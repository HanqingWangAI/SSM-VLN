import numpy as np

import agent_v6
import torch
import torch.optim as optim

import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from optimizer import GlobalAdam
from collections import defaultdict

from tensorboardX import SummaryWriter

from param import args
from env import R2RBatch
import time
import pickle
import traceback
import os

from single_process import EvalProcess, StepProcess, AgentSingleProcess


class Learner:
    def __init__(self, env: dict, results_path, tok, episode_len=20, max_node=40, process_num=16, visible_gpu="0"): # env should be a dictionary of environments
        self.env = env
        self.global_model = agent_v6.SSM(None, results_path, tok, episode_len, max_node, args)
        self.global_model.share_memory()
        self.global_optimizer = GlobalAdam(self.global_model.parameters, args.lr)
        
        for key in self.env:
            bs = self.env[key].env.batch_size
            break

        self.batch_size = bs

        self.global_logs = {}  # a dict of Queue
        self.global_args = {}
        self.global_args['rollout_cnt'] = mp.Value('l', 0)
        self.global_args['iters'] = mp.Value('l', 0)
        self.global_args['log_keys'] = ['select_loss','losses_ml']
        for key in self.global_args['log_keys']:
            self.global_logs[key] = mp.Queue()

        self.grad_Q = mp.Queue()
        
        self.process_num = process_num
        self.visible_gpu = visible_gpu.split(',')
        self.gpu_num = len(self.visible_gpu)
        self.processes = []
        self.sub_envs = {}
        for key in self.env:
            self.sub_envs[key] = self.env[key].split(self.process_num)

        self.sub_envs_args = defaultdict(list)
        for key in self.sub_envs:
            for env in self.sub_envs[key]:
                self.sub_envs_args[key].append((env.env.features, env.data, env.scans, env.env.batch_size))

        self.sub_envs_args = [{key: self.sub_envs_args[key][i] for key in self.sub_envs_args}
                                    for i in range(self.process_num)]
    
    
    def load(self, path):
        self.global_model.load(path)
    
    def load_eval(self, path, part=False):
        self.global_model.load_eval(path, part)

    def train(self, start=0):
        self.global_args['iters'].value = start
        pstep = StepProcess(self.global_model, self.global_optimizer, self.global_logs, self.global_args, self.batch_size, self.grad_Q, -1, 0)
        self.processes.append(pstep)
        for idx in range(self.process_num):
            print('process',idx)
            gpu_id = (idx % (self.gpu_num-1)) + 1
            p1 = AgentSingleProcess(self.sub_envs_args[idx], self.global_model, self.global_optimizer, self.global_logs, self.global_args, self.batch_size, self.grad_Q, idx, gpu_id,'teacher')

            p2 = AgentSingleProcess(self.sub_envs_args[idx], self.global_model, self.global_optimizer, self.global_logs, self.global_args, self.batch_size, self.grad_Q, idx, gpu_id,'sample')

            self.processes.append(p1)
            self.processes.append(p2)

        for p in self.processes:
            p.start()

        for p in self.processes:
            p.join()

        self.processes = []

        print('over')
    
    def eval_init(self):
        self.processes = []
        M = Manager()
        self.res_Q = M.Queue()
        self.sync_Q = []
        for idx in range(self.process_num):
            
            print('process',idx)
            syncq = M.Queue()
            p = EvalProcess(self.sub_envs_args[idx],self.global_model, self.res_Q, syncq, idx)

            self.sync_Q.append(syncq)
            self.processes.append(p)
        
        for p in self.processes:
            p.start()
    
    def eval(self):
        for q in self.sync_Q:
            q.put(True)

        results = defaultdict(list)
        num = self.process_num * len(self.env)
        
        for _ in range(num):
            name, res = self.res_Q.get()
            results[name] += res
            # print('name',name)
        
        torch.cuda.empty_cache()

        return results


  
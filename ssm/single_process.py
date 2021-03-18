import numpy as np

import agent_v6
import torch
import torch.optim as optim

import torch.nn.functional as F
import torch.multiprocessing as mp
from optimizer import GlobalAdam
from collections import defaultdict

from tensorboardX import SummaryWriter

# from param import args
from env import R2RBatch
import time
import pickle
import traceback
import os

class EvalProcess(mp.Process):
    def __init__(self, env_args: dict, master_model, res_Q, sync_Q, process_id=0, gpu_id=0):
        super(EvalProcess, self).__init__(name = "Process-%d" % process_id)
        self.env_args = env_args
        self.master_model = master_model
        self.res_Q = res_Q
        self.sync_Q = sync_Q
        self.gpu_id = gpu_id

    def _sync_local_with_global(self): # load the state will not change the gradient of the parameters
        for m_local, m_global in zip(self.model.models, self.master_model.models):
            m_local.load_state_dict(m_global.state_dict())

    def go(self):
        self.envs = {}
        for key in self.env_args:
            # print('env', key)
            feature_store, data, scans, bs = self.env_args[key]
            env = R2RBatch(feature_store, bs, splits=None, tokenizer=self.master_model.tok, name='sub_train',record_scans=scans)
           
            env.data = data
            self.envs[key] = env
            k = key

        
        while True:
            _ = self.sync_Q.get()
            self.model = agent_v6.SSM(self.envs[k], self.master_model.results_path, self.master_model.tok, self.master_model.episode_len, self.master_model.max_node, self.master_model.args)
            self._sync_local_with_global()

            for model in self.model.models:
                model.eval()

            for name in self.envs:
                # print('doing', name)
                iters = None if name != 'train' else 20
                # iters = 1
                self.model.env = self.envs[name]
                self.model.test(use_dropout=False, feedback='argmax', iters=iters)

                res = self.model.get_results()
                self.res_Q.put((name, res))
            
            del self.model
            self.model = None
            torch.cuda.empty_cache()
    
    def run(self):
        self.go()
        print('finished tasks')

class StepProcess(mp.Process):
    def __init__(self, master_model, global_optimizer, global_logs, global_args, batch_size, grad_Q, process_id=0, gpu_id=0):
        super(StepProcess, self).__init__(name = "Process-%d" % process_id)
        self.global_optimizer = global_optimizer
        self.master_model = master_model
        self.global_logs = global_logs
        self.global_args = global_args
        self.batch_size = batch_size
        self.grad_Q = grad_Q
        self.gpu_id = gpu_id

        self.log_keys = self.global_args['log_keys']
    
    def _update_global_model(self):
        for model in self.master_model.models:
            for params in model.parameters():
                if params.grad is not None:
                    # print('have gradient')
                    params.grad /= self.global_args['rollout_cnt'].value
        
        self.master_model.clip_grad()
        self.global_optimizer.step()
        self.global_optimizer.zero_grad()
    
    def _accumulate_global_gradient(self):
        bs, grads_all = self.grad_Q.get()
        self.global_args['rollout_cnt'].value += bs
        for mg_local, m_global in zip(grads_all, self.master_model.models):
            for global_param, g_local in zip(m_global.parameters(),
                                                mg_local):
                if g_local is None:
                    continue
                
                if global_param.grad is not None:
                    global_param.grad = global_param.grad + g_local.cuda(self.gpu_id)
                else:
                    global_param.grad = g_local.cuda(self.gpu_id)

    def step(self):
        while True:
            # time.sleep(0.5)
            self._accumulate_global_gradient()

            if self.global_args['rollout_cnt'].value >= self.batch_size:
                self._update_global_model()

                self.global_args['rollout_cnt'].value = 0
                self.global_args['iters'].value += 1
                idx = self.global_args['iters'].value
                

                if idx % 20 == 0:
                    # write logs
                    str_ = ''
                    for key in self.log_keys:
                        q = self.global_logs[key]
                        c = []
                        while not q.empty():
                            c += q.get()

                        val = sum(c)/len(c)
                        self.writer.add_scalar("loss/%s"%key, val, idx)
                        str_ += "loss/%s: %f "%(key,val)

                    print('iter', idx, str_)
                
                if idx % 100 == 0:
                    path = os.path.join("snap", self.master_model.args.name, "state_dict", "Iter_%06d" % (idx))
                    self.master_model.save(idx, path)

    def run(self):
        torch.cuda.set_device(self.gpu_id)
        # create logs
        logdir = 'snap/%s' % self.master_model.args.name
        self.writer = SummaryWriter(logdir=logdir)
        print('writer path',logdir)
        print('arguements')
        print(self.master_model.args)

        self.step()
                
class AgentSingleProcess(mp.Process):
    def __init__(self, env_args: dict, master_model, global_optimizer, global_logs, global_args: dict, batch_size: int, grad_Q: mp.Queue, process_id=0, gpu_id=0, feedback=None): 
        ''' env_args is a dictionary of the environment arguments {env_name: args}
        '''
        super(AgentSingleProcess, self).__init__(name = "Process-%d" % process_id)
        # NOTE: self.master.* refers to parameters shared across all processes
        # NOTE: self.*        refers to process-specific properties
        # NOTE: we are not copying self.master.* to self.* to keep the code clean

        self.global_optimizer = global_optimizer
        self.global_logs = global_logs
        self.master_model = master_model
        self.process_id = process_id
        self.grad_Q = grad_Q
        self.gpu_id = gpu_id
        assert feedback in ['teacher','sample']
        self.feedback = feedback

        
        self.env_args = env_args

        self.batch_size = batch_size
        self.global_args = global_args
        print(self.global_args)
        self.log_keys = self.global_args['log_keys']
        self.version = -1
        
    def _sync_local_with_global(self): # load the state will not change the gradient of the parameters
        while self.version == int(self.global_args['iters'].value):
            # if self.process_id == 0 and p_flag:
            #     print('process',self.process_id,'wait','current ver',self.version)
            #     p_flag = False
            time.sleep(0.1)
        for m_local, m_global in zip(self.model.models, self.master_model.models):
            m_local.load_state_dict(m_global.state_dict())
        
        self.optimizer.zero_grad()
        self.version = int(self.global_args['iters'].value)

    def _ensure_global_grads(self):
        for m_local, m_global in zip(self.model.models, self.master_model.models):
            for global_param, local_param in zip(m_global.parameters(),
                                                m_local.parameters()):
                if global_param.grad is not None:
                    return
                global_param._grad = local_param.grad
    
    def _accumulate_global_gradient(self):
        for m_local, m_global in zip(self.model.models, self.master_model.models):
            for global_param, local_param in zip(m_global.parameters(),
                                                m_local.parameters()):
                if local_param.grad is None:
                    # print(m_local.__class__)
                    continue
                # else:
                #     print('yes!')
                if global_param.grad is not None:
                    global_param.grad = global_param.grad + local_param.grad
                else:
                    global_param._grad = local_param.grad
    
    def _send_grad(self, bs):
        grads_all = []
        for m_local in self.model.models:
            grad_m = []
            for local_param in m_local.parameters():
                if local_param.grad is None:
                    grad_m.append(None)
                    continue
                grad_m.append(local_param.grad.clone().detach())
            grads_all.append(grad_m)

        self.grad_Q.put((bs, grads_all))

    def _check_version(self):
        # return True
        return self.global_args['iters'].value == self.version

    def go(self):
        self.envs = {}
        for key in self.env_args:
            feature_store, data, scans, bs = self.env_args[key]
            env = R2RBatch(feature_store, bs, splits=None, tokenizer=self.master_model.tok, name='sub_train',record_scans=scans)
           
            env.data = data   
            self.envs[key] = env

        self.model = agent_v6.SSM(self.envs['train'], self.master_model.results_path, self.master_model.tok, self.master_model.episode_len, self.master_model.max_node, self.master_model.args)
        
        self.optimizer = optim.Adam(self.model.parameters, self.master_model.args.lr)


        for model in self.model.models:
            model.train()

        try:
            for idx in range(self.master_model.args.iters):
                # print('iter',idx, self.master_model.args.iters)
                self._sync_local_with_global() # sync the local model with the global model
               
                self.model.logs = defaultdict(list)

                if 'aug' in self.envs:
                    if (idx + self.gpu_id) % 2 == 0: 
                        self.model.env = self.envs['train']
                        self.master_model.args.ml_weight = 0.2
                    else:
                        self.model.env = self.envs['aug']
                        self.master_model.args.ml_weight = 0.6
                else:
                    self.model.env = self.envs['train']
                    self.master_model.args.ml_weight = 0.2

                self.model.loss = 0.

                self.model.feedback = self.feedback
                self.model.rollout(train_ml=self.master_model.args.ml_weight, train_rl=False)

                self.model.loss.backward()

                for key in self.log_keys:
                    self.global_logs[key].put(self.model.logs[key])

                self._send_grad(int(bs/2)) # send gradient to the update process

                

        except Exception as ex:
            traceback.print_exc()
 
    def run(self):
        torch.cuda.set_device(self.gpu_id)
        print('enter', os.environ["CUDA_VISIBLE_DEVICES"])
        self.go()
              





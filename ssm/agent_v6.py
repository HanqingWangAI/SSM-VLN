import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer
import utils
import model
import param
# from param import args
from collections import defaultdict
from copy import copy, deepcopy
from torch import multiprocessing as mp
# from mp import Queue
# from torch.multiprocessing import Queue
from torch.multiprocessing import Process, Queue
# import imp
# imp.reload(model)
from speaker import Speaker
from graph import GraphBatch

from sklearn.svm import SVC
from model import check

class SF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return (input>0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def obs_process(batch):
    res = []
    for item in batch:
        res.append({
            'instr_id' : item['instr_id'],
            'scan' : item['scan'],
            'viewpoint' : item['viewpoint'],
            'viewIndex' : item['viewIndex'],
            'heading' : item['heading'],
            'elevation' : item['elevation'],
            # 'navigableLocations' : item['navigableLocations'],
            'instructions' : item['instructions'],
            'path_id' : item['path_id']
        })
    return res

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                # print('iter',i)
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break




class SSM(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    def __init__(self, env, results_path, tok, episode_len=20, max_node=40, global_args=None):
        super(SSM, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        if self.env is None:
            self.feature_size = 2048
        else:
            self.feature_size = self.env.feature_size
        self.v_size = self.feature_size
        if global_args is not None:
            self.args = global_args
        else:
            raise NameError("Need the argument")
        
        # self.queue = Queue()
        # self.queue = Queue()
        if not env is None:
            self.gb = GraphBatch(self.env.batch_size, self.feature_size, v_size=self.feature_size, max_node=max_node,args=self.args)
        self.max_node = max_node

        # Models
        enc_hidden_size = self.args.rnn_dim//2 if self.args.bidir else self.args.rnn_dim
        self.encoder = model.EncoderLSTM(tok.vocab_size(), self.args.wemb, enc_hidden_size, padding_idx,
                                         self.args.dropout, bidirectional=self.args.bidir,sub_out=self.args.sub_out,zero_init=self.args.zero_init).cuda()

        self.critic = model.Critic().cuda()
        self.critic_exp = model.Critic().cuda()

        ld = {}
        ld['attnuv'] = model.AttnUV(self.args.rnn_dim, self.args.rnn_dim, self.args.dropout, feature_size=self.feature_size + self.args.angle_feat_size,featdropout=self.args.featdropout,angle_feat_size=self.args.angle_feat_size).cuda()
        ld['linear_vt'] = model.FullyConnected(self.feature_size + self.args.angle_feat_size, self.v_size).cuda()
        ld['linear_ot'] = model.FullyConnected(4, self.args.angle_feat_size).cuda()
        ld['linear_ha'] = model.FullyConnected(self.args.rnn_dim, 4, True).cuda()
        ld['gru_a'] = nn.GRUCell(self.args.angle_feat_size, self.args.angle_feat_size).cuda()
        ld['gru_p'] = nn.GRUCell(self.v_size, self.v_size).cuda()


        ld['selector'] = model.DecoderLSTM(self.args.aemb, self.args.rnn_dim, self.args.dropout, ld['attnuv'], feature_size=self.v_size + self.args.angle_feat_size,featdropout=self.args.featdropout,angle_feat_size=self.args.angle_feat_size).cuda()

        ld['decoder'] = model.Decoder(self.args.rnn_dim, self.args.dropout, feature_size=self.feature_size + self.args.angle_feat_size, angle_feat_size=self.args.angle_feat_size).cuda()

        self.package = model.ModulePackage(ld)
        

        self.models = (self.encoder, self.critic, self.critic_exp, self.package)
        self.models_part = (self.encoder, self.critic)

        self.parameters = []
        for m in self.models:
            self.parameters.append({'params': m.parameters()})
        

        # Optimizers
        self.encoder_optimizer = self.args.optimizer(self.encoder.parameters(), lr=self.args.lr * 0.05)
        self.critic_optimizer = self.args.optimizer(self.critic.parameters(), lr=self.args.lr)


        self.critic_exp_optimizer = self.args.optimizer(self.critic_exp.parameters(), lr=self.args.lr)

        self.package_optimizer = self.args.optimizer(self.package.parameters(), lr=self.args.lr)

        self.optimizers = (self.encoder_optimizer, self.critic_optimizer, self.critic_exp_optimizer, self.package_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False, reduce=False)
        
        self.criterion_gate = nn.CrossEntropyLoss(ignore_index=2, size_average=False, reduce=False)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        # print('Initialization finished')

    def share_memory(self):
        for m in self.models:
            m.share_memory()

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]    # seq_lengths[0] is the Maximum length

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.bool().cuda(),  \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), self.args.views, self.feature_size + self.args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return torch.from_numpy(features).float().cuda()

    # @profile
    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + self.args.angle_feat_size), dtype=np.float32) # [batch, max_candidat_length, feature_size]
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_viewpoint(self, obs):
        viewpoints = []
        for i, ob in enumerate(obs):
            viewpoints.append(ob['viewpoint'])
        return viewpoints

    # @profile
    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended, stop=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if not stop is None:
                    a[i] = len(ob['candidate'])
                    continue
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()
    
    def _teacher_action_candidate(self, perm_obs, vids, perm_idx, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        goals = []
        for item in self.env.batch:
            goals.append(item['path'][-1])
        
        perm_goals = [goals[idx] for idx in perm_idx]

        a = np.zeros(len(perm_obs), dtype=np.int64)
        for i, ob in enumerate(perm_obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:

                scan = ob['scan']
                vp = ob['viewpoint']
                name_list = vids[i]
                goal = perm_goals[i]

                if vp == goal:
                    a[i] = len(name_list)
                else:
                    distances = [self.env.distances[scan][_][goal] for _ in name_list]

                    if len(distances) == 0:
                        a[i] = 0
                    else:
                        a[i] = np.argmin(distances)

        return torch.from_numpy(a).cuda()
        

    def _teacher_front(self, perm_obs, names, perm_idx, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        goals = []
        for item in self.env.batch:
            goals.append(item['path'][-1])
        
        perm_goals = [goals[idx] for idx in perm_idx]

        a = np.zeros(len(perm_obs), dtype=np.int64)

        for i, ob in enumerate(perm_obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                scan = ob['scan']
                name_list = names[i]
                goal = perm_goals[i]
                if len(name_list) == 0:
                    a[i] = self.args.ignoreid
                    continue
                distances = [self.env.distances[scan][_][goal] for _ in name_list]
                a[i] = np.argmin(distances)


        return torch.from_numpy(a).cuda()

    
    def make_equiv_action(self, env, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view 
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                env.env.sims[idx].makeAction(*self.env_actions[name])
            state = env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                # print(action, len(perm_obs[i]['candidate']))
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def make_equiv_action_name(self, a_t, perm_obs, candidate_name, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view 
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                # print('action',action,'length',len(candidate_name[i]))
                n = candidate_name[i][action]
                for _ in perm_obs[i]['candidate']:
                    if _['viewpointId'] == n:
                        select_candidate = _
                        break
                # select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def make_reward(self, cpu_a_t_after, cpu_a_t_before, perm_idx):
        obs = np.array(self.env._get_obs())
        scanIds = [ob['scan'] for ob in obs]
        viewpoints = [ob['viewpoint'] for ob in obs]
        headings = [ob['heading'] for ob in obs]
        elevations = [ob['elevation'] for ob in obs]
        perm_obs = obs[perm_idx]

        self.make_equiv_action(self.env, cpu_a_t_after, perm_obs, perm_idx)
        obs_temp = np.array(self.env._get_obs())
        perm_obs_temp = obs_temp[perm_idx]

        dist_after = np.array([ob['distance'] for ob in perm_obs_temp])

        self.env.env.newEpisodes(scanIds,viewpoints,headings,elevations)

        self.make_equiv_action(self.env, cpu_a_t_before, perm_obs, perm_idx)
        obs_temp = np.array(self.env._get_obs())
        perm_obs_temp = obs_temp[perm_idx]

        dist_before = np.array([ob['distance'] for ob in perm_obs_temp])

        self.env.env.newEpisodes(scanIds,viewpoints,headings,elevations)
        reward = (dist_before > dist_after).astype(np.float32) - (dist_before < dist_after).astype(np.float32) + 3 * ((dist_after < 3).astype(np.float32) - (dist_after > 3).astype(np.float32)) * (cpu_a_t_after == -1).astype(np.float32) - 0.1
        # return torch.from_numpy(reward).cuda().float()
        return reward

    # @profile
    def update_state(self, ctx, ctx_mask, h, ended, noise=None):
        v_f, _, _, num_list = self.gb.get_nodes()
        v_f = torch.from_numpy(v_f).float().cuda()
        # s_f = torch.from_numpy(s_f).float().cuda()
        batch_size = len(num_list)

        # v_tilde, u = self.attnuv(v_f, h, num_list, ctx_padding, ctx_mask_padding, noise)
        v_tilde, hp, ha = self.package.attnuv(v_f, h, num_list, ctx, ctx_mask, noise)

        ha = ha.reshape(batch_size, 1, 1, -1) # batch x 1 x 1 x dim

        # hu = torch.cat([h,u], -1) # batch x dim
        # hu = hu.reshape(batch_size, 1, 1, -1) # batch x 1 x 1 x dim

        v_tilde = self.package.linear_vt(v_tilde) # batch x num x v_size
        # print('v_tilde',check(v_tilde), 'u', check(u))

        v_tilde_list = []
        # u_list = []
        # s_list = []
        # m_list = []
        cnt = 0

        for i, n in enumerate(num_list):
            v_ = v_tilde[cnt : cnt + n]
            # u_ = u[i].repeat(n,1) # n x dim
            cnt += n
            v_tilde_list.append(v_)
            # u_list.append(u_)

        
        v_tilde = self.feature_padding(v_tilde_list, self.max_node) # batch x num x dim
        # u_padding = self.feature_padding(u_list, self.max_node) # batch x num x dim


        o = self.gb.get_edges()
        o = torch.from_numpy(o).float().cuda() # batch x num x num x 4

        o_tilde = ((self.package.linear_ha(ha) * o).sum(-1).unsqueeze(-1) * self.package.linear_ot(o)).sum(2) # batch x num x dim

        A = (o.sum(-1) != 0).float()
        
        v = v_tilde
        o = o_tilde
        # print('v',v.shape,'o',o.shape,'A',A.shape)
        # message passing
        for _ in range(3):
            v_sum = torch.bmm(A, v).reshape(-1,self.v_size)
            o_sum = torch.bmm(A, o).reshape(-1,self.args.angle_feat_size)
            v = self.package.gru_p(v_sum, v.reshape(-1,self.v_size))
            o = self.package.gru_a(o_sum, o.reshape(-1,self.args.angle_feat_size))
            v = v.reshape(batch_size, -1, self.v_size)
            o = o.reshape(batch_size, -1, self.args.angle_feat_size)
            # v = torch.tanh(torch.bmm(A_v, self.package.linear_v(v)))
            # o = torch.tanh(torch.bmm(A_a, self.package.linear_o(o)))
        
        self.gb.update_states(v, o, ended)

        # if check(s):
        #     print('s is nan',s)
        # else:
        #     print('correct')

        return v, o


    def jump(self, names, perm_idx, obs, traj, ended, ctx, ctx_mask, h1, c1, noise):
        '''
            obs should be the obs before permutation
        '''
        batch_size = len(obs)
        perm_obs = obs[perm_idx]

        scanIds = [ob['scan'] for ob in perm_obs]
        viewpoints = [ob['viewpoint'] for ob in perm_obs]
        # headings = [ob['heading'] for ob in perm_obs]
        # elevations = [ob['elevation'] for ob in perm_obs]

        paths = []
        for i, (vp, tp) in enumerate(zip(viewpoints, names)):
            path = self.gb.graphs[i].get_path(vp,tp)
            paths.append(path[1:])


        ended = np.array(ended)

        h1_res = h1
        c1_res = c1
        ha_res = torch.zeros_like(h1)
        hp_res = torch.zeros_like(h1)
        cnt = 0

        while True:
            if ended.all():
                break
            
            a = np.ones(batch_size).astype(np.int32) * -1
            for i, path in enumerate(paths):
                if cnt >= len(path):
                    a[i] = -1
                else:
                    ob = perm_obs[i]
                    cs = ob['candidate']
                    for j, c in enumerate(cs):
                        if c['viewpointId'] == path[cnt]:
                            a[i] = j
                            break

                    if a[i] == -1:
                        # print(path[cnt],ob['viewpoint'],ended[i])
                        assert path[cnt] == ob['viewpoint']
            
            cnt += 1
            self.make_equiv_action(self.env, a, perm_obs,perm_idx, traj)

            pre_obs = perm_obs
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu
            
            self.gb.add_nodes(perm_obs, pre_obs, ended) # update the graph

            s, m = self.update_state(ctx, ctx_mask, h1, ended, noise) # batch x num x dim

            idx = self.gb.get_index(perm_obs)

            input_a_t = self.angle_feature(perm_obs)

            input_f = torch.zeros(batch_size, self.v_size + self.args.angle_feat_size).cuda()

            for i, id_ in enumerate(idx):
                input_f[i,:self.v_size] = s[i, id_]
                input_f[i,self.v_size:] = m[i, id_]

            h1, c1, hp, ha = self.package.selector(input_a_t, input_f, None,
                h1, c1,
                ctx, ctx_mask)
            
            mask = torch.from_numpy(ended).cuda().reshape(-1,1).float()
            
            h1_res =  h1_res * mask + h1 * (1.0-mask)
            c1_res =  c1_res * mask + c1 * (1.0-mask)
            ha_res =  ha_res * mask + ha * (1.0-mask)
            hp_res =  hp_res * mask + hp * (1.0-mask)
            
            ended = (a == -1) | ended

        # print('cnt',cnt)
            

        return h1_res, c1_res, hp_res, ha_res

    def feature_padding(self, f, max_num = 50):
        '''
        The input should be a list of tensors.
        Padding zeros to f, to make its shape become batch x max_num x dim
        '''
        res = []
        for _ in f:
            shape = _.shape
            pad = torch.zeros(max_num-shape[0],shape[1]).cuda()
            res.append(torch.cat([_,pad],0))
        
        return torch.stack(res, 0) # batch x max_num x dim
    
    def angle_feature(self, obs):
        heading = np.array([ob['heading'] for ob in obs])
        elevation = np.array([ob['elevation'] for ob in obs])
        
        heading = torch.from_numpy(heading).cuda()
        elevation = torch.from_numpy(elevation).cuda()
        feat = torch.stack([torch.sin(heading), torch.cos(heading),
                torch.sin(elevation), torch.cos(elevation)
            ],-1).repeat(1, self.args.angle_feat_size // 4).float()

        return feat

    # @profile
    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        # print('step in')
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # print('step in2')
        noise = self.package.decoder.drop_env(torch.ones(self.feature_size).cuda())
        if speaker is not None:         # Trigger the self_train mode!
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))
        

        # Reorder the language input for the encoder (do not ruin the original code)
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        self.gb.start(perm_obs)

        # print('before encoder')
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx_mask = seq_mask
        # print('after encoder')

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # For test result submission
        visited = [set() for _ in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        policy_log_probs_front = []
        masks = []
        entropys = []
        entropys_front = []
        ml_loss_list = []


        ml_loss = 0.
        select_loss = 0.
        navigate_loss = 0.


        rl_loss_exp = 0.

        
        h1 = h_t
        c1 = c_t
        traj_length = np.zeros(batch_size).astype(np.int32)
        
    

        for t in range(self.episode_len):
            # print('t',t)
            # print('average node',sum([len(g.dict) for g in self.gb.graphs])/batch_size)

            # #################################
            #
            #         update the graph
            #         select node and jump
            #
            # #################################
            # print('ended',ended)
            # pos = [ob['viewpoint'] for ob in perm_obs]
            # print('now',pos)
            s, m = self.update_state(ctx, ctx_mask, h1, ended, noise) # batch x num x dim

            idx = self.gb.get_index(perm_obs)

            input_a_t = self.angle_feature(perm_obs)

            input_f = torch.zeros(batch_size, self.v_size + self.args.angle_feat_size).cuda()

            for i, id_ in enumerate(idx):
                input_f[i,:self.v_size] = s[i, id_]
                input_f[i,self.v_size:] = m[i, id_]

            names_front, frontiers = self.gb.get_frontier_nodes()
            # names_front, frontiers = self.gb.get_all_nodes()

            frontier_leng = [len(c) for c in frontiers]
            front_feat = np.zeros((batch_size, max(frontier_leng), self.v_size + self.args.angle_feat_size), dtype=np.float32)

            front_feat = torch.from_numpy(front_feat).float().cuda()
            for i, fronts in enumerate(frontiers):
                for j, front in enumerate(fronts):
                    s_, m_ = front
                    front_feat[i,j,:self.v_size] = s_
                    front_feat[i,j,self.v_size:] = m_

            h1, c1, logit_front, hp, ha = self.package.selector(input_a_t, input_f, front_feat,
            h1, c1,
            ctx, ctx_mask)

            front_mask = utils.length2mask(frontier_leng)
            logit_front.masked_fill_(front_mask, -float('inf'))

            target = self._teacher_front(perm_obs, names_front, perm_idx, ended)
            select_loss += (self.criterion(logit_front, target) * torch.from_numpy(~ended).float().cuda()).sum()

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t_front = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _, a_t_front = logit_front.max(1)        # student forcing - argmax
                a_t_front = a_t_front.detach()
                log_probs = F.log_softmax(logit_front, 1)                              # Calculate the log_prob here
                policy_log_probs_front.append(log_probs.gather(1, a_t_front.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample' or self.feedback == 'teacher':
                probs = F.softmax(logit_front, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys_front.append(c.entropy())                                # For optimization
                a_t_front = c.sample().detach()
                policy_log_probs_front.append(c.log_prob(a_t_front))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            cpu_a_t_front = a_t_front.detach().cpu().numpy()

            for i, next_id in enumerate(cpu_a_t_front):
                if next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t_front[i] = -1             # Change the <end> and ignore action to -1
            
            names_perm = []
            for i, c in enumerate(cpu_a_t_front):
                if c == -1:
                    names_perm.append(perm_obs[i]['viewpoint'])
                else:
                    names_perm.append(names_front[i][c])
            
            h1, c1, hp, ha = self.jump(names_perm, perm_idx, obs, traj, ended, ctx, ctx_mask, h1, c1, noise) # jump to the selected viewpoint

            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]

            candidates = self.gb.get_local(perm_obs)
            candidate_leng = [len(c) + 1 for c in candidates]

            cand_feat = np.zeros((batch_size, max(candidate_leng), self.feature_size + self.args.angle_feat_size), dtype=np.float32)
            candidate_name = []

            for i, cand in enumerate(candidates):
                names = []
                for j, c in enumerate(cand): # c: <v, o>
                    # print(c)
                    cand_feat[i, j, :] = c[2][0]   # Image feat
                    names.append(c[1])

                candidate_name.append(names)
            
            cand_feat = torch.from_numpy(cand_feat).float().cuda()

            logit = self.package.decoder(cand_feat, hp, ha, ctx, ctx_mask, noise)
            
            hidden_states.append(h1.detach())


            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))
            # print('after_final',logit)


            # Supervised training
            target = self._teacher_action_candidate(perm_obs, candidate_name, perm_idx, ended)
            navigate_loss += (self.criterion(logit, target) * torch.from_numpy(~ended).float().cuda()).sum()
            

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)

                # self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                # print(a_t)
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            
            # Make action and get the new state
            # self.make_equiv_action(self.env, cpu_a_t, perm_obs, perm_idx, traj)
            self.make_equiv_action_name(cpu_a_t, perm_obs, candidate_name, perm_idx, traj)
            pre_obs = perm_obs
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu
            
            self.gb.add_nodes(perm_obs, pre_obs, ended) # update the graph

            # self.logs['graph'].append(deepcopy(self.gb.graphs[0].G))
            # Calculate the mask and reward
            dist = np.zeros(batch_size, np.float32)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            for i, ob in enumerate(perm_obs):
                dist[i] = ob['distance']
                if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                    reward[i] = 0.
                    mask[i] = 0.
                else:       # Calculate the reward
                    action_idx = cpu_a_t[i]
                    if action_idx == -1:        # If the action now is end
                        if dist[i] < 3:         # Correct
                            reward[i] = 2.
                        else:                   # Incorrect
                            reward[i] = -2.
                    else:                       # The action is not end
                        reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                        if reward[i] > 0:                           # Quantification
                            reward[i] = 1
                        elif reward[i] < 0:
                            reward[i] = -1
                        # else:
                        #     raise NameError("The action doesn't change the move")

            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            
            traj_length += (t+1) * np.logical_and(ended == 0, (cpu_a_t == -1))

            flag = self.gb.dis_in_range(perm_obs)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1) & flag)
     
            # Early exit if all ended
            if ended.all(): 
                break
        

        traj_length += self.episode_len * (ended == 0)

        if True:
            s, m = self.update_state(ctx, ctx_mask, h1, ended, noise) # batch x num x dim

            idx = self.gb.get_index(perm_obs)

            input_a_t = self.angle_feature(perm_obs)

            input_f = torch.zeros(batch_size, self.v_size + self.args.angle_feat_size).cuda()

            for i, id_ in enumerate(idx):
                input_f[i,:self.v_size] = s[i, id_]
                input_f[i,self.v_size:] = m[i, id_]

            names_front, frontiers = self.gb.get_nodes_in_range()
            # names_front, frontiers = self.gb.get_all_nodes()

            frontier_leng = [len(c)+1 for c in frontiers]
            front_feat = np.zeros((batch_size, max(frontier_leng), self.v_size + self.args.angle_feat_size), dtype=np.float32)

            front_feat = torch.from_numpy(front_feat).float().cuda()
            for i, fronts in enumerate(frontiers):
                for j, front in enumerate(fronts):
                    s_, m_ = front
                    front_feat[i,j,:self.v_size] = s_
                    front_feat[i,j,self.v_size:] = m_

            h1, c1, logit_front, hp, ha = self.package.selector(input_a_t, input_f, front_feat,
            h1, c1,
            ctx, ctx_mask)

            
            front_mask = utils.length2mask(frontier_leng)
            logit_front.masked_fill_(front_mask, -float('inf'))

            target = self._teacher_front(perm_obs, names_front, perm_idx, ended)
            select_loss += (self.criterion(logit_front, target) * torch.from_numpy(~ended).float().cuda()).sum()

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t_front = target                # teacher forcing
            elif self.feedback == 'argmax': 
                logit_front = logit_front.clone()
                for i,c in enumerate(frontier_leng):
                    logit_front[i,c-1] = -10000000
                _, a_t_front = logit_front.max(1)        # student forcing - argmax
                a_t_front = a_t_front.detach()
                log_probs = F.log_softmax(logit_front, 1)                              # Calculate the log_prob here
                policy_log_probs_front.append(log_probs.gather(1, a_t_front.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample' or self.feedback == 'teacher':
                probs = F.softmax(logit_front, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys_front.append(c.entropy())                                # For optimization
                a_t_front = c.sample().detach()
                policy_log_probs_front.append(c.log_prob(a_t_front))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            cpu_a_t_front = a_t_front.detach().cpu().numpy()

            for i, next_id in enumerate(cpu_a_t_front):
                if next_id == self.args.ignoreid or ended[i] or next_id == frontier_leng[i]-1:    # The last action is <end>
                    cpu_a_t_front[i] = -1             # Change the <end> and ignore action to -1
            
            names_perm = []
            for i, c in enumerate(cpu_a_t_front):
                if c == -1:
                    names_perm.append(perm_obs[i]['viewpoint'])
                else:
                    names_perm.append(names_front[i][c])
            
            h1, c1, hp, ha = self.jump(names_perm, perm_idx, obs, traj, ended, ctx, ctx_mask, h1, c1, noise) # jump to the selected viewpoint


        

        if train_rl:
            # Last action in A2C
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            if speaker is not None:
                candidate_feat[..., :-self.args.angle_feat_size] *= noise
                f_t[..., :-self.args.angle_feat_size] *= noise
            last_h_, _, _, _,_,_ = self.decoder(input_a_t, f_t, candidate_feat,
                                            h1, c_t,
                                            ctx, ctx_mask,
                                            speaker is not None)
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
    
        if train_ml is not None:
            # print('explore_rate', explore_rate)
            ml_loss = navigate_loss + select_loss
            self.loss += ml_loss * train_ml / batch_size
            self.logs['select_loss'].append(select_loss.item()/batch_size)
            self.logs['losses_ml'].append((ml_loss * train_ml / batch_size).item() / self.episode_len)
        
        # if exp_il:
        #     # ml_loss_policy = ml_loss_policy / batch_size
        #     policy_entropy = policy_entropy / batch_size
        #     self.loss += policy_entropy
        
        # self.loss += rl_loss_exp
  
        if type(rl_loss_exp) is float or type(rl_loss_exp) is int:
            self.logs['rl_loss_exp'].append(0.)
        else:
            self.logs['rl_loss_exp'].append(rl_loss_exp.item())


        if type(self.loss) is int or type(self.loss) is float:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.
        self.logs['traj'].append(traj)
        return traj



    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, **kwargs):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            # self.encoder.train()
            # self.decoder.train()
            # self.linear.train()
            # self.explorer.train()
            # self.policy.train()
            # self.critic.train()
            # self.critic_exp.train()
            # self.critic_policy.train()
            for module in self.models:
                module.train()
            
        else:
            # self.encoder.eval()
            # self.decoder.eval()
            # self.linear.eval()
            # self.explorer.eval()
            # self.policy.eval()
            # self.critic.eval()
            # self.critic_exp.eval()
            # self.critic_policy.eval()
            for module in self.models:
                module.eval()

        with torch.no_grad():
            super(SSM, self).test(iters, **kwargs)
    
    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.explorer.parameters(), 40.)
        

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.explorer_optimizer.step()
        self.critic_optimizer.step()
        self.critic_exp_optimizer.step()

    def clip_grad(self):
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        # torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
        # torch.nn.utils.clip_grad_norm(self.explorer.parameters(), 40.)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer),
                     ("critic_exp", self.critic_exp, self.critic_exp_optimizer),
                     ("package", self.package, self.package_optimizer)
                     ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN",name)
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if self.args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                    ("critic", self.critic, self.critic_optimizer),
                    ("critic_exp", self.critic_exp, self.critic_exp_optimizer)
                    ]

        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

    def load_eval(self, path, part=False):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if self.args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        if part:
            all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                        ("critic", self.critic, self.critic_optimizer)
                        ]
        else:
            all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer),
                     ("critic_exp", self.critic_exp, self.critic_exp_optimizer),
                     ("package", self.package, self.package_optimizer)
                    ]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1


''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
sys.path.append('../build')
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args

from utils import load_datasets, load_nav_graphs, Tokenizer
import threading
import queue
csv.field_size_limit(sys.maxsize)
from copy import copy, deepcopy
from multiprocessing import Process, Queue


import torch
# class data_fetch_worker(object):
#     def __init__(self, io)

class ViewPoint():
    def __init__(self,location):
        self.viewpointId = location.viewpointId
        self.ix = location.ix
        # self.x = location.x
        # self.y = location.y
        # self.z = location.z
        self.rel_heading = location.rel_heading
        self.rel_elevation = location.rel_elevation
        self.rel_distance = location.rel_distance


class State():

    def __init__(self, state):
        self.scanId = state.scanId
        self.location = ViewPoint(state.location)
        self.heading = state.heading
        self.elevation = state.elevation
        self.nevigableLocations = self.navigableLocations(state.navigableLocations)
        self.viewIndex = state.viewIndex

    def navigableLocations(self, locations):
        res = []
        for v in locations:
            res.append(ViewPoint(v))
        
        return res



class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                # print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        self.batch_size = batch_size
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings, elevations=None):
        if elevations is None:
            elevations = np.zeros(len(headings))
        for i, (scanId, viewpointId, heading, elevation) in enumerate(zip(scanIds, viewpointIds, headings, elevations)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, elevation)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states


    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)
    
    def copystate(self, env):
        for i, sim in enumerate(self.sims):
            state = env.sims[i].getState()
            scanId = state.scanId
            viewpointId = state.location.viewpointId
            heading = state.heading
            elevation = state.elevation
            sim.newEpisode(scanId, viewpointId, heading, elevation)

    def copyinstance(self):
        env = EnvBatch(self.features, len(self.sims))
        for i, sim in enumerate(env.sims):
            state = self.sims[i].getState()
            scanId = state.scanId
            viewpointId = state.location.viewpointId
            heading = state.heading
            elevation = state.elevation
            sim.newEpisode(scanId, viewpointId, heading, elevation)
        
        return env
        
class EnvBatch_P():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features 
        Parallel version
        '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
   

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                # print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        self.qin = []
        self.qout = []
        self.qtraj = []
        self.feature_states = None
        self.batch_size = batch_size
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

            self.qin.append(Queue())
            self.qout.append(Queue())
            self.qtraj.append(Queue())
        
        self.pool = []

        def function(i, sim, qin, qout, qtraj):
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
            while True:
                op, x = qin.get()
                
                # while not qout.empty():
                #     qout.get()
                # while not qtraj.empty():
                #     qtraj.get()

                if op == 'act':
                    select_candidate, src_point, trg_point, src_level, trg_level = x
                    traj = []
                    def take_action(i, name):
                        if type(name) is int:       # Go to the next view
                            sim.makeAction(name, 0, 0)
                        else:                       # Adjust
                            sim.makeAction(*env_actions[name])
                        state = sim.getState()
                        
                        traj.append((state.location.viewpointId, state.heading, state.elevation))
                        

                    while src_level < trg_level:    # Tune up
                        take_action(i, 'up')
                        src_level += 1
                    while src_level > trg_level:    # Tune down
                        take_action(i, 'down')
                        src_level -= 1
                    while sim.getState().viewIndex != trg_point:    # Turn right until the target
                        take_action(i, 'right')
                    assert select_candidate['viewpointId'] == \
                        sim.getState().navigableLocations[select_candidate['idx']].viewpointId
                    take_action(i, select_candidate['idx'])

                    state = sim.getState()
                    qout.put((state.scanId, state.location.viewpointId, state.heading, state.elevation))
                    # print(traj)
                    qtraj.put(traj)

                elif op == 'new':
                    scanId, viewpointId, heading, elevation = x
                    sim.newEpisode(scanId, viewpointId, heading, elevation)

                elif op == 'state':
                    state = sim.getState()
                    state = State(state)
                        # if name != 'env':
                        #     setattr(env_copy,name,value)
                    
                    qout.put(state)

        for i in range(batch_size):
            p = Process(target=function, args=(i, self.sims[i], self.qin[i], self.qout[i], self.qtraj[i]), daemon=True)
            
            p.start()
            self.pool.append(p)
        


    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings, elevations=None):
        if elevations is None:
            elevations = np.zeros(len(headings))
        for i, (scanId, viewpointId, heading, elevation) in enumerate(zip(scanIds, viewpointIds, headings, elevations)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.qin[i].put(('new',(scanId, viewpointId, heading, elevation)))
            # self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

        self.feature_states = self._getStates()
  
    def _getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        # for i, sim in enumerate(self.sims):
        #     state = sim.getState()

        #     long_id = self._make_id(state.scanId, state.location.viewpointId)
        #     if self.features:
        #         feature = self.features[long_id]     # Get feature for
        #         feature_states.append((feature, state))
        #     else:
        #         feature_states.append((None, state))
        for i in range(self.batch_size):
            while not self.qout[i].empty():
                self.qout[i].get()
            while not self.qtraj[i].empty():
                self.qtraj[i].get()

            self.qin[i].put(('state',None))
        
        for i in range(self.batch_size):
            state = self.qout[i].get()
            # print(state)
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))

        return feature_states

    def getStates(self):
        if self.feature_states is None:
            self.feature_states = self._getStates()
        
        return self.feature_states


    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        pool = []
        def makeaction(i, env, index, heading, elevation, q):
            env.makeAction(index, heading, elevation)
            state = env.getState()
            
            scanId = state.scanId
            viewpointId = state.location.viewpointId
            heading = state.heading
            elevation = state.elevation
            q.put((i,scanId,viewpointId,heading,elevation))
            

        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)
            p = Process(target=makeaction, args=(i,self.sims[i],index,heading,elevation,self.q))
            p.start()
            pool.append(p)
        
        for p in pool:
            p.join()
        
        while not self.q.empty():
            i, scanId, viewpointId, heading, elevation = self.q.get()
            self.sims[i].newEpisode(scanId, viewpointId, heading, elevation)

    
    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        
        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                
                self.qin[idx].put(('act',(select_candidate, src_point, trg_point, src_level, trg_level)))
                
        
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:
                if traj is not None:
                    res = self.qtraj[i].get()
                    traj[i]['path'] += res

                scanId, viewpointId, heading, elevation = self.qout[i].get()

        #         idx = perm_idx[i]
        #         self.sims[idx].newEpisode(scanId, viewpointId, heading, elevation)

        self.feature_states = self._getStates()

    def copystate(self, env):
        for i, sim in enumerate(self.sims):
            state = env.sims[i].getState()
            scanId = state.scanId
            viewpointId = state.location.viewpointId
            heading = state.heading
            elevation = state.elevation
            sim.newEpisode(scanId, viewpointId, heading, elevation)

    def copyinstance(self):
        env = EnvBatch(self.features, len(self.sims))
        for i, sim in enumerate(env.sims):
            state = self.sims[i].getState()
            scanId = state.scanId
            viewpointId = state.location.viewpointId
            heading = state.heading
            elevation = state.elevation
            sim.newEpisode(scanId, viewpointId, heading, elevation)
        
        return env
     
class EnvBatch_T():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features 
        Parallel version
        '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
   

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                # print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        self.qin = []
        self.qout = []
        self.qtraj = []
        self.feature_states = None
        self.batch_size = batch_size
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

            self.qin.append(queue.Queue())
            self.qout.append(queue.Queue())
            self.qtraj.append(queue.Queue())
        
        self.pool = []

        def function(i, sim, qin, qout, qtraj):
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
            while True:
                op, x = qin.get()
                
                # while not qout.empty():
                #     qout.get()
                # while not qtraj.empty():
                #     qtraj.get()

                if op == 'act':
                    select_candidate, src_point, trg_point, src_level, trg_level = x
                    traj = []
                    def take_action(i, name):
                        if type(name) is int:       # Go to the next view
                            sim.makeAction(name, 0, 0)
                        else:                       # Adjust
                            sim.makeAction(*env_actions[name])
                        state = sim.getState()
                        
                        traj.append((state.location.viewpointId, state.heading, state.elevation))
                        

                    while src_level < trg_level:    # Tune up
                        take_action(i, 'up')
                        src_level += 1
                    while src_level > trg_level:    # Tune down
                        take_action(i, 'down')
                        src_level -= 1
                    while sim.getState().viewIndex != trg_point:    # Turn right until the target
                        take_action(i, 'right')
                    assert select_candidate['viewpointId'] == \
                        sim.getState().navigableLocations[select_candidate['idx']].viewpointId
                    take_action(i, select_candidate['idx'])

                    state = sim.getState()
                    qout.put((state.scanId, state.location.viewpointId, state.heading, state.elevation))
                    # print(traj)
                    qtraj.put(traj)

                elif op == 'new':
                    scanId, viewpointId, heading, elevation = x
                    sim.newEpisode(scanId, viewpointId, heading, elevation)

                elif op == 'state':
                    state = sim.getState()
                    state = State(state)
                        # if name != 'env':
                        #     setattr(env_copy,name,value)
                    
                    qout.put(state)

        for i in range(batch_size):
            p = threading.Thread(target=function, args=(i, self.sims[i], self.qin[i], self.qout[i], self.qtraj[i]), daemon=True)
            
            p.start()
            self.pool.append(p)
        


    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings, elevations=None):
        if elevations is None:
            elevations = np.zeros(len(headings))
        for i, (scanId, viewpointId, heading, elevation) in enumerate(zip(scanIds, viewpointIds, headings, elevations)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.qin[i].put(('new',(scanId, viewpointId, heading, elevation)))
            # self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

        self.feature_states = self._getStates()
  
    def _getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        # for i, sim in enumerate(self.sims):
        #     state = sim.getState()

        #     long_id = self._make_id(state.scanId, state.location.viewpointId)
        #     if self.features:
        #         feature = self.features[long_id]     # Get feature for
        #         feature_states.append((feature, state))
        #     else:
        #         feature_states.append((None, state))
        for i in range(self.batch_size):
            while not self.qout[i].empty():
                self.qout[i].get()
            while not self.qtraj[i].empty():
                self.qtraj[i].get()

            self.qin[i].put(('state',None))
        
        for i in range(self.batch_size):
            state = self.qout[i].get()
            # print(state)
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))

        return feature_states

    def getStates(self):
        if self.feature_states is None:
            self.feature_states = self._getStates()
        
        return self.feature_states


    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        pool = []
        def makeaction(i, env, index, heading, elevation, q):
            env.makeAction(index, heading, elevation)
            state = env.getState()
            
            scanId = state.scanId
            viewpointId = state.location.viewpointId
            heading = state.heading
            elevation = state.elevation
            q.put((i,scanId,viewpointId,heading,elevation))
            

        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)
            p = Process(target=makeaction, args=(i,self.sims[i],index,heading,elevation,self.q))
            p.start()
            pool.append(p)
        
        for p in pool:
            p.join()
        
        while not self.q.empty():
            i, scanId, viewpointId, heading, elevation = self.q.get()
            self.sims[i].newEpisode(scanId, viewpointId, heading, elevation)

    
    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        
        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                
                self.qin[idx].put(('act',(select_candidate, src_point, trg_point, src_level, trg_level)))
                
        
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:
                if traj is not None:
                    res = self.qtraj[i].get()
                    traj[i]['path'] += res

                scanId, viewpointId, heading, elevation = self.qout[i].get()

        #         idx = perm_idx[i]
        #         self.sims[idx].newEpisode(scanId, viewpointId, heading, elevation)

        self.feature_states = self._getStates()

    def copystate(self, env):
        for i, sim in enumerate(self.sims):
            state = env.sims[i].getState()
            scanId = state.scanId
            viewpointId = state.location.viewpointId
            heading = state.heading
            elevation = state.elevation
            sim.newEpisode(scanId, viewpointId, heading, elevation)

    def copyinstance(self):
        env = EnvBatch(self.features, len(self.sims))
        for i, sim in enumerate(env.sims):
            state = self.sims[i].getState()
            scanId = state.scanId
            viewpointId = state.location.viewpointId
            heading = state.heading
            elevation = state.elevation
            sim.newEpisode(scanId, viewpointId, heading, elevation)
        
        return env
     


class EnvBatchRGB():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        
        self.features = None
        self.image_w = 320
        self.image_h = 320
        self.vfov = 60
        # self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            # sim.setRenderingEnabled(True)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

        print('finished')

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def getRGB(self):
        """
        get RGB
        """
        rgb_list = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()
            img = Image.fromarray(np.array(state.rgb).astype(np.uint8))
            img = np.array(img)
            img = img[...,[2,1,0]]
            rgb_list.append(img)

        return rgb_list

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class EnvBatchGraph():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId, angleId):
        return scanId + '_' + viewpointId + '_' + angleId  

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [([64,2051]*36, sim_state)] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()
            features = []
            for j in range(36):
                long_id = self._make_id(state.scanId, state.location.viewpointId,str(j+1))
                feature = self.features[long_id]
                pad_num = 64-len(feature)
                
                if pad_num > 0: # padding the feature to [64, 2051]
                    padding = np.zeros([pad_num, 2051])         
                    feature = np.concatenate((feature,padding))

                features.append(feature)
            
            feature_states.append((features, state))
            # if self.features:
            #     feature = self.features[long_id]     # Get feature for
            #     feature_states.append((feature, state))
            # else:
            #     feature_states.append((None, state))

        return feature_states  # [([64,2051]*36), sim_state] * batch_size

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None, record_scans=None):
        
        if feature_store is None:
            return

        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)

        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        if splits is not None:
            for split in splits:
                for item in load_datasets([split]):
                    # Split multiple instructions into separate entries
                    if item['instructions'] == '':
                        new_item = dict(item)
                        new_item['instr_id'] = '%s_%d' % (item['path_id'], 0)
                        new_item['instructions'] = ''
                        self.data.append(new_item)
                        scans.append(item['scan'])
                    else:
                        for j,instr in enumerate(item['instructions']):
                            if item['scan'] not in self.env.featurized_scans:   # For fast training
                                continue
                            new_item = dict(item)
                            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                            new_item['instructions'] = instr
                            if 'seg' in item:
                                
                                if str(j) in item['seg']:
                                    # print(j)
                                    new_item['seg'] = item['seg'][str(j)]
                                else:
                                    continue

                            if tokenizer:
                                new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                            if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                                self.data.append(new_item)
                                scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name
        
        if record_scans is not None:
            scans = record_scans
        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}
        self.buffered_state_dict_detail = {}
        self.batch = None
        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        if splits is not None:
            print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(['None'] if splits is None else splits)))
    
    def split(self, num):
        sub_batch_size = int(self.env.batch_size / num)
        sub_envs = []
        n = list([int(len(self.data) * 1.0 * i / num) for i in range(num+1)])
        data_split = []
        for i in range(num):
            data_split.append(self.data[n[i]:n[i+1]])
        # [self.data[n[i]:n[i+1]] for i in range(num)]
        feature_store = self.env.features
        for i in range(num):
            env = R2RBatch(feature_store, sub_batch_size, seed= self.seed, splits=None, tokenizer=self.tok,  name=self.name)
            env.data = data_split[i]
            env.scans = self.scans
            sub_envs.append(env)
        
        return sub_envs
        

    def copystate(self, env):
        self.env.copystate(env.env)
        for name, value in vars(env).items():
            if name != 'env':
                setattr(self,name,value)

    def copyinstance(self):
        env_copy = R2RBatch(None)
        for name, value in vars(self).items():
            if name != 'env':
                setattr(env_copy,name,value)
        
        setattr(env_copy,'env',self.env.copyinstance())
        return env_copy

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        # print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        # if shuffle:
        #     random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    # @profile
    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        long_id_detail = '%s_%s'%(long_id, viewId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            self.buffered_state_dict_detail[long_id_detail] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx','heading','feature']}
                for c in candidate
            ]
            return candidate
        elif long_id_detail not in self.buffered_state_dict_detail:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                # c_new.pop('normalized_heading')
                candidate_new.append(c_new)

            self.buffered_state_dict_detail[long_id_detail] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx','heading','feature']}
                for c in candidate_new
            ]
            
            return candidate_new
        else:
            return self.buffered_state_dict_detail[long_id_detail]

    # @profile
    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex
            # print('in 1')
            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            
            # print('in 2')

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id'],
                'seg': item['seg'] if 'seg' in item else None
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        # print('in 3')
        return obs

    def _get_obs_fake(self,dest):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):

            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'teacher' : self._shortest_path_action(state, dest),

            })

        return obs

    def _get_path_length(self):
        length = []
        for item in self.batch:
            path = self.paths[item['scan']][item['path'][0]][item['path'][-1]]
            length.append(len(path))
        return length

    def _get_progress(self, obs):
        res = []
        vps = [ob['viewpoint'] for ob in obs]
        for i, item in enumerate(self.batch):
            v = vps[i]
            a = len(self.paths[item['scan']][item['path'][0]][v])
            b = len(self.paths[item['scan']][v][item['path'][-1]])
            res.append(1.0*a/(a+b-1))
        
        return np.array(res)

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]  
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()
    
    def reset_fake(self, batch=None, inject=False, **kwargs):
        # ''' Load a new minibatch / episodes. '''
        if self.batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        # else:
        #     if inject:          # Inject the batch into the next minibatch
        #         self._next_minibatch(**kwargs)
        #         self.batch[:len(batch)] = batch
        #     else:               # Else set the batch to the current batch
        #         self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats

class R2RBatch_preload():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        
        if feature_store is None:
            return

        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)

        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.q = queue.Queue(1)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}
        self.batch = None
        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

        def fetch_data():
            while True:
                batch = self._getbatch()
                self.q.put(batch)
                # print('finish')

        self.th = threading.Thread(target=fetch_data)
        self.th.start()
        self.one_batch = None

    def copystate(self, env):
        self.env.copystate(env.env)
        for name, value in vars(env).items():
            if name != 'env':
                setattr(self,name,value)

    def copyinstance(self):
        env_copy = R2RBatch(None)
        for name, value in vars(self).items():
            if name != 'env':
                setattr(env_copy,name,value)
        
        setattr(env_copy,'env',self.env.copyinstance())
        return env_copy

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        # if shuffle:
        #     random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        # print(state.scanId, state.location.viewpointId, goalViewpointId)
        # if not goalViewpointId in self.paths[state.scanId][state.location.viewpointId]:
        #     print(state.scanId, state.location.viewpointId, goalViewpointId, item)
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex
            if item['scan'] != state.scanId:
                scans_a = [state[1].scanId for state in self.env.getStates()]
                scans_b = [item['scan'] for item in self.batch]
                scans_ab = [(a,b) for a,b in zip(scans_a,scans_b)]
                print(scans_ab)

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()
    
    def reset_fake(self, batch=None, inject=False, **kwargs):
        # ''' Load a new minibatch / episodes. '''
        if self.batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        # else:
        #     if inject:          # Inject the batch into the next minibatch
        #         self._next_minibatch(**kwargs)
        #         self.batch[:len(batch)] = batch
        #     else:               # Else set the batch to the current batch
        #         self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


    def make_equiv_action(self, a_t, perm_obs):
        """
        Interface between Panoramic view and Egocentric view 
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
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
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.sims[idx].makeAction(*env_actions[name])
            
        perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action

                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12

                # print(src_level, trg_level, self.env.sims[idx].getState().viewIndex, trg_point)
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                # print('yes')
                while self.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print(self.env.sims[idx].getState().viewIndex)

                assert select_candidate['viewpointId'] == \
                       self.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])


    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return a
    
    def _getbatch(self):
        obs = np.array(self.reset())
        ended = np.array([False] * self.batch_size)  # Indices match permuation of the model, not env
        length = 10
        obs_list = []
        # obs_list.append(obs)
        exp_list = []
        action_list = []

        for t in range(length):
            candidate_batch = []  
            mask_batch = [] 
            scanIds = [ob['scan'] for ob in obs]
            viewpoints = [ob['viewpoint'] for ob in obs]
            headings = [ob['heading'] for ob in obs]
            elevations = [ob['elevation'] for ob in obs]

            candidate_leng = [len(ob['candidate']) for ob in obs]
            max_length = max(candidate_leng)

            for i in range(max_length):
                a = np.zeros(len(obs), dtype=np.int64)
                mask = np.zeros(len(obs), dtype=np.int64)
                for j, ob in enumerate(obs):
                    if i >= len(ob['candidate']):
                        a[j] = -1
                    else:
                        a[j] = i
                        mask[j] = 1
                
                self.make_equiv_action(a, obs)
                obs_cand = np.array(self._get_obs())
                candidate_batch.append(obs_cand)
                mask_batch.append(mask)
                self.env.newEpisodes(scanIds,viewpoints,headings,elevations)
                
            candidate_batch = np.array(candidate_batch).transpose() # batch x max_cand_length (obs)
            mask_batch = np.array(mask_batch).transpose() # batch x max_cand_length (mask)


            a_t = self._teacher_action(obs, ended)
            obs_list.append(obs)
            exp_list.append((candidate_batch, mask_batch))
            action_list.append(np.array(a_t))

            for i, next_id in enumerate(a_t):
                if next_id == candidate_leng[i] or next_id == args.ignoreid:    # The last action is <end>
                    a_t[i] = -1             # Change the <end> and ignore action to -1

            if ended.all(): 
                break

            
            self.make_equiv_action(a_t, obs)
            obs = np.array(self._get_obs())
            
            

            ended[:] = np.logical_or(ended, (a_t == -1))
            
            

        # assert len(obs_list) == len(exp_list)
        return obs_list, exp_list, action_list


    def getbatch(self):
        res = self.q.get()
        return res


    def getbatch_fake(self):
        if self.one_batch is None:
            self.one_batch = self.q.get()
        # res = self.q.get()
        res = self.one_batch
        return res

class R2RBatch_preload_P():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        
        if feature_store is None:
            return

        self.env = EnvBatch_T(feature_store=feature_store, batch_size=batch_size)

        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.q = queue.Queue(100)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}
        self.batch = None
        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

        def fetch_data():
            while True:
                batch = self._getbatch()
                self.q.put(batch)
                # print('finish')

        self.th = threading.Thread(target=fetch_data)
        self.th.start()

    def copystate(self, env):
        self.env.copystate(env.env)
        for name, value in vars(env).items():
            if name != 'env':
                setattr(self,name,value)

    def copyinstance(self):
        env_copy = R2RBatch(None)
        for name, value in vars(self).items():
            if name != 'env':
                setattr(env_copy,name,value)
        
        setattr(env_copy,'env',self.env.copyinstance())
        return env_copy

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        # if shuffle:
        #     random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()
    
    def reset_fake(self, batch=None, inject=False, **kwargs):
        # ''' Load a new minibatch / episodes. '''
        if self.batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        # else:
        #     if inject:          # Inject the batch into the next minibatch
        #         self._next_minibatch(**kwargs)
        #         self.batch[:len(batch)] = batch
        #     else:               # Else set the batch to the current batch
        #         self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


    def make_equiv_action(self, a_t, perm_obs):
        """
        Interface between Panoramic view and Egocentric view 
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
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
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.sims[idx].makeAction(*env_actions[name])
            
        perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action

                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12

                # print(src_level, trg_level, self.env.sims[idx].getState().viewIndex, trg_point)
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                # print('yes')
                while self.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print(self.env.sims[idx].getState().viewIndex)

                assert select_candidate['viewpointId'] == \
                       self.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])


    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return a
    
    def _getbatch(self):
        obs = np.array(self.reset())
        ended = np.array([False] * self.batch_size)  # Indices match permuation of the model, not env
        length = 10
        obs_list = []
        # obs_list.append(obs)
        exp_list = []
        action_list = []

        for t in range(length):
            candidate_batch = []  
            mask_batch = [] 
            scanIds = [ob['scan'] for ob in obs]
            viewpoints = [ob['viewpoint'] for ob in obs]
            headings = [ob['heading'] for ob in obs]
            elevations = [ob['elevation'] for ob in obs]

            candidate_leng = [len(ob['candidate']) for ob in obs]
            max_length = max(candidate_leng)

            for i in range(max_length):
                a = np.zeros(len(obs), dtype=np.int64)
                mask = np.zeros(len(obs), dtype=np.int64)
                for j, ob in enumerate(obs):
                    if i >= len(ob['candidate']):
                        a[j] = -1
                    else:
                        a[j] = i
                        mask[j] = 1
                
                self.env.make_equiv_action(a, obs)
                obs_cand = np.array(self._get_obs())
                candidate_batch.append(obs_cand)
                mask_batch.append(mask)
                self.env.newEpisodes(scanIds,viewpoints,headings,elevations)
                
            candidate_batch = np.array(candidate_batch).transpose() # batch x max_cand_length (obs)
            mask_batch = np.array(mask_batch).transpose() # batch x max_cand_length (mask)


            a_t = self._teacher_action(obs, ended)
            obs_list.append(obs)
            exp_list.append((candidate_batch, mask_batch))
            action_list.append(np.array(a_t))

            for i, next_id in enumerate(a_t):
                if next_id == candidate_leng[i] or next_id == args.ignoreid:    # The last action is <end>
                    a_t[i] = -1             # Change the <end> and ignore action to -1

            if ended.all(): 
                break

            
            self.env.make_equiv_action(a_t, obs)
            obs = np.array(self._get_obs())
            
            

            ended[:] = np.logical_or(ended, (a_t == -1))
            
            

        # assert len(obs_list) == len(exp_list)
        return obs_list, exp_list, action_list


    def getbatch(self):
        res = self.q.get()
        return res


class R2RBatch_P():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        
        if feature_store is None:
            return

        self.env = EnvBatch_P(feature_store=feature_store, batch_size=batch_size)

        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}
        self.batch = None
        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))
    
    def copystate(self, env):
        self.env.copystate(env.env)
        for name, value in vars(env).items():
            if name != 'env':
                setattr(self,name,value)

    def copyinstance(self):
        env_copy = R2RBatch(None)
        for name, value in vars(self).items():
            if name != 'env':
                setattr(env_copy,name,value)
        
        setattr(env_copy,'env',self.env.copyinstance())
        return env_copy

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        # if shuffle:
        #     random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()
    
    def reset_fake(self, batch=None, inject=False, **kwargs):
        # ''' Load a new minibatch / episodes. '''
        if self.batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        # else:
        #     if inject:          # Inject the batch into the next minibatch
        #         self._next_minibatch(**kwargs)
        #         self.batch[:len(batch)] = batch
        #     else:               # Else set the batch to the current batch
        #         self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


class R2RBatch_aug():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            path = item['path']
            vp =  state.location.viewpointId
            nxp = None
            for j,p in enumerate(path[1:]):
                if path[j] == vp:
                    nxp = p
                    break

            if nxp is None:
                nxp = path[-1]
                
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, nxp),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats

class R2RBatch_neg():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []

        self.scan_specific_data = {}
    
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        if not item['scan'] in self.scan_specific_data:
                            self.scan_specific_data[item['scan']] = []
                        
                        self.scan_specific_data[item['scan']].append(new_item)
                        scans.append(item['scan'])

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        # self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch_neg loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = deepcopy(batch)
        
        self.start_list = []
        self.dest_list = []
        self.fake_start_list = []
        self.fake_dest_list = []

        
        for i,item in enumerate(self.batch):
            self.start_list.append(item['path'][0])
            self.dest_list.append(item['path'][-1])
            path_length = len(item['path'])
            scan = item['scan']
            fake_flag = True
            fail_flag = False
            goal_list = [goal for goal in self.paths[scan][self.start_list[-1]]]
            random.shuffle(goal_list)

            for goal in goal_list:
                if abs(path_length - len(self.paths[scan][self.start_list[-1]][goal])) < 1 and self.distances[scan][self.dest_list[-1]][goal] > 3:
                    self.fake_dest_list.append(self.paths[scan][self.start_list[-1]][goal])
                    # print('fake_dest',i)
                    fake_flag = False
                    break

            if fake_flag:
                fail_flag = True
                # print('fake dest error')
                self.fake_dest_list.append(item['path'])

            fake_flag = True
            goal_list = [goal for goal in self.paths[scan][self.dest_list[-1]]]
            random.shuffle(goal_list)

            for goal in goal_list:
                if abs(path_length - len(self.paths[scan][self.dest_list[-1]][goal])) < 1 and self.distances[scan][self.start_list[-1]][goal] > 3:
                    self.fake_start_list.append(self.paths[scan][self.dest_list[-1]][goal])
                    fake_flag = False
                    break
            
            
            if fake_flag:
                fail_flag = True
                # print('fake start error')
                self.fake_start_list.append(item['path'])

            # print('scan',scan)

            if i != 0 and fail_flag:
                self.batch[i] = deepcopy(self.batch[i-1])
                self.start_list[-1] = self.start_list[-2]
                self.dest_list[-1] = self.dest_list[-2]
                self.fake_start_list[-1] = self.fake_start_list[-2]
                self.fake_dest_list[-1] = self.fake_dest_list[-2]
            

            # cnt_dest = 0
            # cnt_star = 0
            # scan = self.batch[i]['scan']
            # item = self.batch[i]
            # # print('scan after',scan)
            # fake_dest_path = self.paths[scan][self.start_list[-1]][self.fake_dest_list[-1]]
            # fake_star_path = self.paths[scan][self.fake_start_list[-1]][self.dest_list[-1]]
            # for p in item['path']:
            #     if p in fake_dest_path:
            #         cnt_dest += 1
            #     if p in fake_star_path:
            #         cnt_star += 1
            # dis_dest = self.distances[scan][item['path'][-1]][self.fake_dest_list[-1]]
            # dis_star = self.distances[scan][item['path'][0]][self.fake_start_list[-1]]
            # print('length',path_length,'fake dest',cnt_dest, 'fake start',cnt_star,'dis:','dest',dis_dest,'start',dis_star)

            # print('ori',item['path'])
            # print('des',self.paths[scan][self.start_list[-1]][self.fake_dest_list[-1]])
            # print('sta',self.paths[scan][self.fake_start_list[-1]][self.dest_list[-1]])
            # print('')


    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        # long_id = "%s_%s" % (scanId, viewpointId)
        # if long_id not in self.buffered_state_dict:
        for ix in range(36):
            if ix == 0:
                self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
            elif ix % 12 == 0:
                self.sim.makeAction(0, 1.0, 1.0)
            else:
                self.sim.makeAction(0, 1.0, 0)

            state = self.sim.getState()
            assert state.viewIndex == ix

            # Heading and elevation for the viewpoint center
            heading = state.heading - base_heading
            elevation = state.elevation

            visual_feat = feature[ix]

            # get adjacent locations
            for j, loc in enumerate(state.navigableLocations[1:]):
                # if a loc is visible from multiple view, use the closest
                # view (in angular distance) as its representation
                distance = _loc_distance(loc)

                # Heading and elevation for for the loc
                loc_heading = heading + loc.rel_heading
                loc_elevation = elevation + loc.rel_elevation
                angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                if (loc.viewpointId not in adj_dict or
                        distance < adj_dict[loc.viewpointId]['distance']):
                    adj_dict[loc.viewpointId] = {
                        'heading': loc_heading,
                        'elevation': loc_elevation,
                        "normalized_heading": state.heading + loc.rel_heading,
                        'scanId':scanId,
                        'viewpointId': loc.viewpointId, # Next viewpoint id
                        'pointId': ix,
                        'distance': distance,
                        'idx': j + 1,
                        'feature': np.concatenate((visual_feat, angle_feat), -1)
                    }
        candidate = list(adj_dict.values())
        # self.buffered_state_dict[long_id] = [
        #     {key: c[key]
        #         for key in
        #         ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
        #             'pointId', 'idx']}
        #     for c in candidate
        # ]
        return candidate
        # else:
        #     candidate = self.buffered_state_dict[long_id]
        #     candidate_new = []
        #     for c in candidate:
        #         c_new = c.copy()
        #         ix = c_new['pointId']
        #         normalized_heading = c_new['normalized_heading']
        #         visual_feat = feature[ix]
        #         loc_heading = normalized_heading - base_heading
        #         c_new['heading'] = loc_heading
        #         angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
        #         c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
        #         c_new.pop('normalized_heading')
        #         candidate_new.append(c_new)
        #     return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            path = item['path']
            vp =  state.location.viewpointId
            nxp = None
            for j,p in enumerate(path[1:]):
                if path[j] == vp:
                    nxp = p
                    break

            if nxp is None:
                nxp = path[-1]

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, nxp),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, type_ ='ps', **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch

        assert type_ in ['ps','rw']

        if type_ == 'rw':
            for i in range(len(self.batch)):
                if i % 2 == 0:
                    self.batch[i]['path'] = self.fake_start_list[i]
                else:
                    self.batch[i]['path'] = self.fake_dest_list[i]

        elif type_ == 'ps':
            ins_list_shuffle = []
            for item in self.batch:
                scan = item['scan']
                ins = item['instructions']
                random.shuffle(self.scan_specific_data[scan])
                for _ in self.scan_specific_data[scan]:
                    if _['instructions'] != ins:
                        case = self.scan_specific_data[scan][0]
                        break
                    
                ins_list_shuffle.append((case['instructions'],case['instr_encoding']))
            
            # random.shuffle(ins_list_shuffle)
            for i in range(len(self.batch)):
                self.batch[i]['instructions'] = ins_list_shuffle[i][0]
                self.batch[i]['instr_encoding'] = ins_list_shuffle[i][1]

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]

            

        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()
    
    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats

class R2RBatch_neg_bk():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []

        self.scan_specific_data = {}
    
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        if not item['scan'] in self.scan_specific_data:
                            self.scan_specific_data[item['scan']] = []
                        
                        self.scan_specific_data[item['scan']].append(new_item)
                        scans.append(item['scan'])

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        # self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch_neg loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = deepcopy(batch)
        
        self.start_list = []
        self.dest_list = []
        self.fake_start_list = []
        self.fake_dest_list = []

        
        for i,item in enumerate(self.batch):
            self.start_list.append(item['path'][0])
            self.dest_list.append(item['path'][-1])
            path_length = len(item['path'])
            scan = item['scan']
            fake_flag = True
            fail_flag = False
            goal_list = [goal for goal in self.paths[scan][self.start_list[-1]]]
            random.shuffle(goal_list)

            for goal in goal_list:
                if abs(path_length - len(self.paths[scan][self.start_list[-1]][goal])) < 1 and self.distances[scan][self.dest_list[-1]][goal] > 3:
                    self.fake_dest_list.append(goal)
                    # print('fake_dest',i)
                    fake_flag = False
                    break

            if fake_flag:
                fail_flag = True
                # print('fake dest error')
                self.fake_dest_list.append(item['path'][-1])

            fake_flag = True
            goal_list = [goal for goal in self.paths[scan][self.dest_list[-1]]]
            random.shuffle(goal_list)

            for goal in goal_list:
                if abs(path_length - len(self.paths[scan][self.dest_list[-1]][goal])) < 1 and self.distances[scan][self.start_list[-1]][goal] > 3:
                    self.fake_start_list.append(goal)
                    fake_flag = False
                    break
            
            
            if fake_flag:
                fail_flag = True
                # print('fake start error')
                self.fake_start_list.append(item['path'][0])

            # print('scan',scan)

            if i != 0 and fail_flag:
                self.batch[i] = deepcopy(self.batch[i-1])
                self.start_list[-1] = self.start_list[-2]
                self.dest_list[-1] = self.dest_list[-2]
                self.fake_start_list[-1] = self.fake_start_list[-2]
                self.fake_dest_list[-1] = self.fake_dest_list[-2]
            

            # cnt_dest = 0
            # cnt_star = 0
            # scan = self.batch[i]['scan']
            # item = self.batch[i]
            # # print('scan after',scan)
            # fake_dest_path = self.paths[scan][self.start_list[-1]][self.fake_dest_list[-1]]
            # fake_star_path = self.paths[scan][self.fake_start_list[-1]][self.dest_list[-1]]
            # for p in item['path']:
            #     if p in fake_dest_path:
            #         cnt_dest += 1
            #     if p in fake_star_path:
            #         cnt_star += 1
            # dis_dest = self.distances[scan][item['path'][-1]][self.fake_dest_list[-1]]
            # dis_star = self.distances[scan][item['path'][0]][self.fake_start_list[-1]]
            # print('length',path_length,'fake dest',cnt_dest, 'fake start',cnt_star,'dis:','dest',dis_dest,'start',dis_star)

            # print('ori',item['path'])
            # print('des',self.paths[scan][self.start_list[-1]][self.fake_dest_list[-1]])
            # print('sta',self.paths[scan][self.fake_start_list[-1]][self.dest_list[-1]])
            # print('')


    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        # long_id = "%s_%s" % (scanId, viewpointId)
        # if long_id not in self.buffered_state_dict:
        for ix in range(36):
            if ix == 0:
                self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
            elif ix % 12 == 0:
                self.sim.makeAction(0, 1.0, 1.0)
            else:
                self.sim.makeAction(0, 1.0, 0)

            state = self.sim.getState()
            assert state.viewIndex == ix

            # Heading and elevation for the viewpoint center
            heading = state.heading - base_heading
            elevation = state.elevation

            visual_feat = feature[ix]

            # get adjacent locations
            for j, loc in enumerate(state.navigableLocations[1:]):
                # if a loc is visible from multiple view, use the closest
                # view (in angular distance) as its representation
                distance = _loc_distance(loc)

                # Heading and elevation for for the loc
                loc_heading = heading + loc.rel_heading
                loc_elevation = elevation + loc.rel_elevation
                angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                if (loc.viewpointId not in adj_dict or
                        distance < adj_dict[loc.viewpointId]['distance']):
                    adj_dict[loc.viewpointId] = {
                        'heading': loc_heading,
                        'elevation': loc_elevation,
                        "normalized_heading": state.heading + loc.rel_heading,
                        'scanId':scanId,
                        'viewpointId': loc.viewpointId, # Next viewpoint id
                        'pointId': ix,
                        'distance': distance,
                        'idx': j + 1,
                        'feature': np.concatenate((visual_feat, angle_feat), -1)
                    }
        candidate = list(adj_dict.values())
        # self.buffered_state_dict[long_id] = [
        #     {key: c[key]
        #         for key in
        #         ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
        #             'pointId', 'idx']}
        #     for c in candidate
        # ]
        return candidate
        # else:
        #     candidate = self.buffered_state_dict[long_id]
        #     candidate_new = []
        #     for c in candidate:
        #         c_new = c.copy()
        #         ix = c_new['pointId']
        #         normalized_heading = c_new['normalized_heading']
        #         visual_feat = feature[ix]
        #         loc_heading = normalized_heading - base_heading
        #         c_new['heading'] = loc_heading
        #         angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
        #         c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
        #         c_new.pop('normalized_heading')
        #         candidate_new.append(c_new)
        #     return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, type_ ='ps', **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch

        assert type_ in ['ps','rw']

        if type_ == 'rw':
            for i in range(len(self.batch)):
                if i % 2 == 0:
                    self.batch[i]['path'] = [self.fake_start_list[i],self.dest_list[i]]
                else:
                    self.batch[i]['path'] = [self.start_list[i],self.fake_dest_list[i]]

        elif type_ == 'ps':
            ins_list_shuffle = []
            for item in self.batch:
                scan = item['scan']
                ins = item['instructions']
                random.shuffle(self.scan_specific_data[scan])
                for _ in self.scan_specific_data[scan]:
                    if _['instructions'] != ins:
                        case = self.scan_specific_data[scan][0]
                        break
                    
                ins_list_shuffle.append((case['instructions'],case['instr_encoding']))
            
            # random.shuffle(ins_list_shuffle)
            for i in range(len(self.batch)):
                self.batch[i]['instructions'] = ins_list_shuffle[i][0]
                self.batch[i]['instr_encoding'] = ins_list_shuffle[i][1]

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]

            

        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()
    
    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


class R2RBatch_graph():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatchGraph(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatchGraph loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            # 'feature': np.concatenate((visual_feat, angle_feat), -1)
                            'feature': visual_feat,
                            'angle_feature': angle_feat
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                # c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new['feature'] = visual_feat
                c_new['angle_feature'] = angle_feat
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            # feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature, # (64,2051)*36
                'candidate': candidate,
                'angle_feature': self.angle_feature[base_view_id], # should be concate to 'feature' after GCN's output
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats




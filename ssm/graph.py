# from param import args
from math import pi
import numpy as np
import torch
import networkx as nx

class Graph:
    '''
        Graph Representation.
    '''

    def __init__(self, max_node=50):
        self.v = [] # 36 x v_size

        self.s = [] # s_size, the visual feature
        self.m = [] # m_size, 256, the geometry feature

        self.e = [] # <i,j,oreintation>
        # self.v_size = v_size
        # self.s_size = s_size
        self.dict = {} # <vid, num>
        self.num = 0 # number of nodes
        self.ghost = {} # <<from_vid, ghost_vid>, v_size>
        self.max_node = max_node
        self.map = np.zeros([self.max_node, self.max_node, 4])
        self.G = nx.Graph()
    
    def reset(self):
        self.v = []
        self.s = []
        self.m = [] 

        self.e = []
        self.dict = {}
        self.ghost = {}
        self.num = 0
        self.map = np.zeros([self.max_node, self.max_node, 4])
        self.G = nx.Graph()
    
    def add_ghost(self, from_vid, ghost_vid, v, o):
        '''
            actually, it is adding edges. we will not delete the edges.
        '''
        key = (from_vid, ghost_vid) 
        # if not ghost_vid in self.dict:
        if (not key in self.ghost) and (not ghost_vid in self.dict):
            self.ghost[key] = (v, o)
        
        if ghost_vid in self.dict:            
            self.G.add_edge(from_vid, ghost_vid, weight=1)


    def add_node(self, v, s, m, to_, from_=None, o=None):
        '''
            v: visual feature
            s: state
            o: <heading, elevation>
        '''
        if to_ in self.dict:
            if not from_ is None:
                b = self.dict[to_]
                a = self.dict[from_]

                h,e = o
                self.map[a,b] = np.array([
                        np.sin(h), np.cos(h),
                        np.sin(e), np.cos(e)
                    ])
                self.map[b,a] = np.array([
                        np.sin(h+pi), np.cos(h+pi),
                        np.sin(e+pi), np.cos(e+pi)
                    ])
                
                self.G.add_edge(from_, to_, weight=1)

                self.paths = dict(nx.all_pairs_dijkstra_path(self.G))

            return

        self.v.append(v)
        self.s.append(s)
        self.m.append(m)
        # print(v.shape, s.shape)
        self.dict[to_] = self.num
        if self.num == 0:
            self.start = to_
        self.G.add_node(to_)
        self.num += 1

        keys = [key for key in self.ghost]
        for key in keys:
            f_,t_ = key
            if t_ == to_: # no longer a ghost node
                a = self.dict[f_]
                b = self.dict[t_]

                v,o = self.ghost[key]
                h,e = o
                self.map[a,b] = np.array([
                    np.sin(h), np.cos(h),
                    np.sin(e), np.cos(e)
                    ])
                self.map[b,a] = np.array([
                        np.sin(h+pi), np.cos(h+pi),
                        np.sin(e+pi), np.cos(e+pi)
                    ])
                self.ghost.pop(key) 
                
        if not from_ is None:
            b = self.dict[to_]
            a = self.dict[from_]

            h,e = o
            self.map[a,b] = np.array([
                    np.sin(h), np.cos(h),
                    np.sin(e), np.cos(e)
                ])
            self.map[b,a] = np.array([
                    np.sin(h+pi), np.cos(h+pi),
                    np.sin(e+pi), np.cos(e+pi)
                ])
            
            self.G.add_edge(from_, to_, weight=1)
            # self.e.append((a, b, o)) # a->b
            # self.e.append((b, a, o+pi)) # b->a

        self.paths = dict(nx.all_pairs_dijkstra_path(self.G))

    def update_state(self, s, m):
        # vid = obs['viewpoint']
        # id_ = self.dict[vid]
        # self.s[id_] = s[id_]
        for i, _ in enumerate(self.s):
            self.s[i] = s[i]
            self.m[i] = m[i]

    def get_nodes(self):
        return self.v, self.s, self.m
    
    def get_edges(self):
        '''
            get the connectivity and the number of the nodes
        '''
        # return self.e, self.num
        return self.map

    def get_frontier_node(self):
        nodes = []
        node_set = set([])
        for key in self.ghost:
            from_, to_ = key
            node_set.add(from_)
        
        for node in node_set:
            d = self.dis_to_start(node)
            if d > 6:
                continue
            idx = self.dict[node]
            nodes.append((node,self.v[idx],self.s[idx],self.m[idx]))

        return nodes
    
    def get_path(self,a,b):
        return self.paths[a][b]

    def get_all_node(self):
        nodes = []
        
        for node in self.dict:
            idx = self.dict[node]
            nodes.append((node,self.v[idx],self.s[idx],self.m[idx]))

        return nodes

    def get_node_in_range(self):
        nodes = []
        
        for node in self.dict:
            d = len(self.paths[node][self.start]) - 1
            if d < 4 or d > 6:
                continue
            idx = self.dict[node]
            nodes.append((node,self.v[idx],self.s[idx],self.m[idx]))

        return nodes
    
    def get_local_ghost(self, vid):
        res = []
        for key in self.ghost:
            from_, to_ = key
            if from_ == vid:
                res.append((from_,to_,self.ghost[key]))

        return res

    def get_global_ghost(self):
        res = []
        for key in self.ghost:
            from_, to_ = key
            if not to_ in self.dict: # get the frontier ghosts
                res.append((from_,to_,self.ghost[key]))

        return res

    def get_state(self, vid):
        idx = self.dict[vid]

        return self.s[idx], self.m[idx]

    def get_index(self, vid):
        return self.dict[vid]

    def dis_to_start(self, vid):
        p = self.paths[vid][self.start]
        return len(p)-1

class GraphBatch:
    '''
        A batch of Graph
    '''    
    def __init__(self, batch_size, feature_size, v_size=1024, max_node=50, args=None):
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.v_size = v_size
        self.max_node = max_node
        self.graphs = []
        self.args = args
        if args is None:
            raise NameError('args should not be None')
        for _ in range(batch_size):
            self.graphs.append(Graph(max_node))

    def start(self, obs):
        '''
            add the first node at the beginning
        '''
        for g in self.graphs:
            g.reset()
        features = np.empty((len(obs), self.args.views, self.feature_size + self.args.angle_feat_size), dtype=np.float32)
        names = []

        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
            n = ob['viewpoint']
            names.append(n)
            for c in ob['candidate']: # add ghost nodes
                heading = c['heading']
                elevation = c['elevation']
                ghost_vid = c['viewpointId']
                v = c['feature']
                from_vid = n
                self.graphs[i].add_ghost(from_vid, ghost_vid, v, (heading, elevation))
        
        for i, n in enumerate(names):
            # self.graphs[i].add_node(features[i],np.zeros([args.rnn_dim]), n)
            self.graphs[i].add_node(features[i], torch.zeros([self.v_size]).cuda(), torch.zeros([self.args.angle_feat_size]).cuda(), n)

        # pass

    def add_nodes(self, obs, pre_obs, ended):
        features = np.empty((len(obs), self.args.views, self.feature_size + self.args.angle_feat_size), dtype=np.float32)
        names = []

        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
            n = ob['viewpoint']
            names.append(n)
            if ended[i]:  # if ended, will not add ghost nodes
                continue
            for c in ob['candidate']: # add ghost nodes
                heading = c['heading']
                elevation = c['elevation']
                ghost_vid = c['viewpointId']
                v = c['feature']
                from_vid = n
                self.graphs[i].add_ghost(from_vid, ghost_vid, v, (heading, elevation))

        
        pre_names = [ob['viewpoint'] for ob in pre_obs]
        candidates = [ob['candidate'] for ob in pre_obs]

        oreintation = []

        for i, n in enumerate(names):
            cand = candidates[i]
            flag = True
            for c in cand:
                if n == c['viewpointId']:
                    heading = c['heading']
                    elevation = c['elevation']
                    oreintation.append((heading, elevation))
                    flag = False
                    break

            if flag:
                oreintation.append((-1,-1))
        
        for i, n in enumerate(names):
            if ended[i]: # if it reaches the end or already finished
                continue

            pn = pre_names[i]
            o = oreintation[i]
            # self.graphs[i].add_node(features[i],np.zeros([args.rnn_dim]), n, pn, o)
            self.graphs[i].add_node(features[i],torch.zeros([self.v_size]).float().cuda(),torch.zeros([self.args.angle_feat_size]).cuda(), n, pn, o)

    def update_states(self, s, m, ended):
        '''
        The shape of s and m must be batch x num x dim
        '''
        for i, g in enumerate(self.graphs):
            if not ended[i]:
                g.update_state(s[i],m[i])

    def get_index(self, obs):
        idx = []
        for i, ob in enumerate(obs):
            n = ob['viewpoint']
            _ = self.graphs[i].get_index(n)
            idx.append(_)
        
        return np.array(idx)

    def get_nodes(self):
        v_f = []
        s_f = []
        m_f = []
        num_list = []
        for g in self.graphs:
            v,s,m = g.get_nodes()
            num_list.append(len(v))
            v_f += v
            s_f += s
            m_f += m
        
        # return np.array(v_f), np.array(s_f), num_list
        return np.array(v_f), torch.stack(s_f,0), torch.stack(m_f,0), num_list

    def get_edges(self):
        maps = []
        for g in self.graphs:
            m = g.get_edges()
            maps.append(m)

        return np.array(maps)
                
    def get_state(self, obs):
        states = []
        ms = []
        for i, ob in enumerate(obs):
            n = ob['viewpoint']
            s,m = self.graphs[i].get_state(n)
            states.append(s)
            ms.append(m)

        return np.array(states), np.array(ms)

    def get_local(self, obs):
        '''
            return: batch x len x <from, to, v>
        '''
        ghost_batch = []
        for i, ob in enumerate(obs):
            n = ob['viewpoint']
            ghosts = self.graphs[i].get_local_ghost(n)
            ghost_batch.append(ghosts)

        return ghost_batch 
    
    def dis_in_range(self, obs):
        res = []
        for i, ob in enumerate(obs):
            n = ob['viewpoint']
            d = self.graphs[i].dis_to_start(n)
            if d >= 4 and d <= 6:
                res.append(True)
            else:
                res.append(False)

        return np.array(res)

    def get_global(self):
        '''
            return: batch x len x <from, to, v>
        '''
        ghost_batch = []
        for g in self.graphs:
            ghosts = g.get_global_ghost()
            ghost_batch.append(ghosts)
        
        return ghost_batch 
        
    def get_frontier_nodes(self):
        states_batch = []
        names_batch = []
        for g in self.graphs:
            names = []
            states = []
            nodes = g.get_frontier_node()
            if len(nodes) == 0:
                nodes = g.get_all_node()
            for node in nodes:
                n,_,s,m = node
                names.append(n)
                states.append((s, m))
            
            states_batch.append(states)
            names_batch.append(names)
        
        return names_batch, states_batch

    def get_all_nodes(self):
        states_batch = []
        names_batch = []
        for g in self.graphs:
            names = []
            states = []
            nodes = g.get_all_node()
            for node in nodes:
                n,_,s,m = node
                names.append(n)
                states.append((s, m))
            
            states_batch.append(states)
            names_batch.append(names)
        
        return names_batch, states_batch

    def get_nodes_in_range(self):
        states_batch = []
        names_batch = []
        for g in self.graphs:
            names = []
            states = []
            nodes = g.get_node_in_range()
            for node in nodes:
                n,_,s,m = node
                names.append(n)
                states.append((s, m))
            
            states_batch.append(states)
            names_batch.append(names)
        
        return names_batch, states_batch
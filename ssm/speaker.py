from numpy.lib.function_base import piecewise
import torch
import numpy as np
# from param import args
import os
import utils
import model
import torch.nn.functional as F
from utils import padding_idx

from torch.autograd import Variable
from model import check
import random

from collections import namedtuple

InferenceState = namedtuple("InferenceState", "prev_inference_state, flat_index, last_word, word_count, score")


def backchain_inference_states(last_inference_state):
    word_indices = []
    inf_state = last_inference_state
    
    while inf_state is not None:
        word_indices.append(inf_state.last_word)
        inf_state = inf_state.prev_inference_state

    return list(reversed(word_indices))

class Speaker():
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

    def __init__(self, env, tok, args):
        self.env = env
        if self.env is None:
            self.feature_size = 2048
        else:
            self.feature_size = self.env.feature_size

        self.tok = tok
        self.tok.finalize()
        self.args = args

        # Model
        print("VOCAB_SIZE", self.tok.vocab_size())
        self.encoder = model.SpeakerEncoder(self.feature_size+self.args.angle_feat_size, self.args.rnn_dim, self.args.dropout, bidirectional=self.args.bidir).cuda()
        self.decoder = model.SpeakerDecoder(self.tok.vocab_size(), self.args.wemb, self.tok.word_to_index['<PAD>'],
                                            self.args.rnn_dim, self.args.dropout).cuda()
        self.encoder_optimizer = self.args.optimizer(self.encoder.parameters(), lr=self.args.lr)
        self.decoder_optimizer = self.args.optimizer(self.decoder.parameters(), lr=self.args.lr)
        self.models = [self.encoder, self.decoder]

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])

        # Will be used in beam search
        self.nonreduced_softmax_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.tok.word_to_index['<PAD>'],
            size_average=False,
            reduce=False
        )

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.bool().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), self.args.views, self.feature_size + self.args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()
    
    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs

        return input_a_t, f_t

    def train(self, iters):
        for i in range(iters):
            self.env.reset()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.teacher_forcing(train=True)
            print('loss',loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch()  # Get the insts of the result
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.tok.shrink(inst)  # Shrink the words
        return path2inst

    def get_insts_coco(self, iters=None, wrapper=(lambda x: x)):
        '''
            return: 
                   path2inst: a dict {ID: [{ 'caption' : str}, ...], ...}
        '''
        # Get the caption for all the data

        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        if iters is None:
            total = self.env.size()
        else:
            total = iters
        print('total',total)
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            with torch.no_grad():
                insts = self.infer_batch()  # Get the insts of the result
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                path_id = str(path_id)
                if path_id not in path2inst:
                    path2inst[path_id] = []
                    inst = self.tok.decode_sentence(self.tok.shrink(inst))  # Shrink the words and decode
                    path2inst[path_id].append({'caption': inst})

        return path2inst

    def valid(self, N=None, *aargs, **kwargs):
        """

        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(*aargs, **kwargs)

        case_set = set()
        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        # N = 1 if self.args.fast_train else None     # Set the iter to 1 if the fast_train (o.w. the problem occurs)
        metrics = np.zeros(3)
        
        if N is not None:
            for i in range(N):
                self.env.reset()
                metrics += np.array(self.teacher_forcing(train=False))
            metrics /= N
        else:
            cnt = 0
            loop = False
            while True:
                cnt += 1
                obs = self.env.reset()
                for ob in obs:
                    instr_id = ob['instr_id']
                    if instr_id in case_set:
                        loop = True
                        break

                    case_set.add(instr_id)

                metrics += np.array(self.teacher_forcing(train=False))
                if loop:
                    break

            metrics /= cnt

        return (path2inst, *metrics)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
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
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12   # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _teacher_action(self, obs, ended, tracker=None):
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
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size + self.args.angle_feat_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['feature'] # Image feat
        return torch.from_numpy(candidate_feat).cuda()

    def from_shortest_path(self, viewpoints=None, get_first_feat=False, creator=None):
        """
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :param creator: [encoder, decoder]
        :return:
        """
        
        obs = self.env._get_obs()
        batch_size = len(obs)
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        if creator is not None:
            weights_reg = 0.
            cnt = 0
            seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
            
            ctx_f, h_t_f, c_t_f = creator[0](seq, seq_lengths)
            inv_idx = [0 for _ in perm_idx]
            for i,_ in enumerate(perm_idx):
                inv_idx[_] = i
            
            ctx_mask = seq_mask[inv_idx]
            ctx_f = ctx_f[inv_idx]
            h_t_f = h_t_f[inv_idx]
            c_t_f = c_t_f[inv_idx]
            
            h1_f = h_t_f

            rand_idx = [_ for _ in range(batch_size)]

            random.shuffle(rand_idx)


        first_feat = np.zeros((len(obs), self.feature_size+self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -self.args.angle_feat_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])

            input_a_t, f_t_pano = self.get_input_feat(obs) # Image features from obs
            

            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action

            candidate_feat = self._candidate_variable(obs, teacher_action)

            if creator is not None:
                f_t_shuffle = f_t_pano[rand_idx]
                h_t_f, c_t_f, h1_f, f_t_pano, weights = creator[1](input_a_t, f_t_pano, f_t_shuffle, h1_f, c_t_f, ctx_f, ctx_mask)
            
                for i,ob in enumerate(obs):
                    a = teacher_action[i]
                    c = ob['candidate'][a]
                    idx = c['pointId']
                    candidate_feat[i, :-self.args.angle_feat_size] = f_t_pano[i, idx, :-self.args.angle_feat_size]


                weights_reg += (weights.mean(1).sum(1) * torch.from_numpy(~ended).float().cuda()).sum()
                cnt += (~ended).astype(np.float).sum()
                
            img_feats.append(f_t_pano)
            can_feats.append(candidate_feat)
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            if creator is not None:
                return (img_feats, can_feats), length, weights_reg / cnt

            return (img_feats, can_feats), length

    def gt_words(self, obs):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        return torch.from_numpy(seq_tensor).cuda()

    def teacher_forcing(self, train=True, features=None, insts=None, for_listener=False, perm_idx=None, creator=None):
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Get Image Input & Encode
        if features is not None:
            # It is used in calulating the speaker score in beam-search
            # assert insts is not None
            obs = np.array(self.env._get_obs())
            if perm_idx is not None:
                obs = obs[perm_idx]
            (img_feats, can_feats), lengths = features
            ctx = self.encoder(can_feats, img_feats, lengths, already_dropfeat=True)
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            if creator is not None:
                (img_feats, can_feats), lengths, weights_reg = self.from_shortest_path(creator=creator)      # Image Feature (from the shortest path)
            else:
                (img_feats, can_feats), lengths = self.from_shortest_path()

            ctx = self.encoder(can_feats, img_feats, lengths)

        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ctx_mask = utils.length2mask(lengths)

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)                                       # Language Feature

        # Decode
        logits, _, _ = self.decoder(insts, ctx, ctx_mask, h_t, c_t)

        # Because the softmax_loss only allow dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1).contiguous()
        loss = self.softmax_loss(
            input  = logits[:, :, :-1],         # -1 for aligning
            target = insts[:, 1:]               # "1:" to ignore the word <BOS>
        )

        if check(loss):
            print('lengths',lengths)
            print('loss is nan',loss)
            # print('logits', logits)
            for i,t in enumerate(insts):
                l = self.softmax_loss(
                    input = logits[i,:,:-1].unsqueeze(0),
                    target = t[1:].unsqueeze(0)
                )
                if check(l):
                    print('case',i)
                    print('inst',t[1:])
                    print('ctx',check(ctx[i]))

                    print('length',lengths[i])
                                   
                    # for j,label in enumerate(t[1:]):
                    #     label = label.item()
                    #     if label != self.tok.word_to_index['<PAD>']:
                    #         print('pos %d, word %s, logit'%(j, self.tok.index_to_word[label]),logits[i,j])

            assert False

        if for_listener:
            inst_mask = insts[:, 1:] != self.tok.word_to_index['<PAD>']
            return self.nonreduced_softmax_loss(
                input  = logits[:, :, :-1],         # -1 for aligning
                target = insts[:, 1:]               # "1:" to ignore the word <BOS>
            ), inst_mask
            

        if train:
            if creator is not None:
                return loss, weights_reg
            return loss
        else:
            # Evaluation
            _, predict = logits.max(dim=1)                                  # BATCH, LENGTH
            gt_mask = (insts != self.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]    # Not pad and equal to gt
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()     # Exclude <BOS>
            sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
            return loss.item(), word_accu, sent_accu

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        (img_feats, can_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)      # Image Feature (from the shortest path)

        # This code block is only used for the featdrop.
        if featdropmask is not None:
            img_feats[..., :-self.args.angle_feat_size] *= featdropmask
            can_feats[..., :-self.args.angle_feat_size] *= featdropmask

        # Encoder
        ctx = self.encoder(can_feats, img_feats, lengths,
                           already_dropfeat=(featdropmask is not None))
        ctx_mask = utils.length2mask(lengths)

        # Decoder
        words = []
        log_probs = []
        masks = []
        hidden_states = []
        entropies = []
        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ended = np.zeros(len(obs), np.bool)
        word = np.ones(len(obs), np.int64) * self.tok.word_to_index['<BOS>']    # First word is <BOS>
        word = torch.from_numpy(word).view(-1, 1).cuda()
        for i in range(self.args.maxDecode):
            # Decode Step
            logits, h_t, c_t = self.decoder(word, ctx, ctx_mask, h_t, c_t)      # Decode, logits: (b, 1, vocab_size)

            # Select the word
            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            logits[:, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                # print('word',word)
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    hidden_states.append(h_t.squeeze())
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    hidden_states.append(h_t.squeeze().detach())
                    entropies.append(m.entropy().detach())
            else:
                values, word = logits.max(1)

            mask = np.ones(batch_size, np.float32)
            for i, _ in enumerate(ended):
                if _:
                    mask[i] = 0.

            masks.append(mask)

            # Append the word
            cpu_word = word.cpu().numpy()
            cpu_word[ended] = self.tok.word_to_index['<PAD>']
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break

        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(hidden_states, 1), torch.stack(entropies, 1), masks
        else:
            return np.stack(words, 1)       # [(b), (b), (b), ...] --> [b, l]

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
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if self.args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

    def load_eval(self, path):
        return self.load(path)

    def beam_search(self, beam_size=3, train=False):
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        
        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        (img_feats, can_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)      # Image Feature (from the shortest path)

        # Encoder
        ctx = self.encoder(can_feats, img_feats, lengths,
                           already_dropfeat=True)
        ctx_mask = utils.length2mask(lengths)

        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()

        
        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [
            [InferenceState(prev_inference_state=None,
                            flat_index=i,
                            last_word=self.tok.word_to_index['<BOS>'],
                            word_count=0,
                            score=0.0)]
            for i in range(batch_size)
        ]

        for t in range(self.args.maxDecode):
            flat_indices = []
            beam_indices = []
            w_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    w_t_list.append(inf_state.last_word)

            # w_t = try_cuda(Variable(torch.LongTensor(w_t_list), requires_grad=False))
            w_t = torch.from_numpy(np.array(w_t_list)).long().cuda()
            # if len(w_t.shape) == 1:
            #     w_t = w_t.unsqueeze(0)

            logit, h_t, c_t = self.decoder(w_t.view(-1, 1), ctx[beam_indices], ctx_mask[beam_indices], h_t[:,flat_indices], c_t[:,flat_indices])

            logit = logit.squeeze(1)

            logit[:, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer


            # h_t,c_t,alpha,logit = self.decoder(w_t.view(-1, 1), h_t[flat_indices], c_t[flat_indices], ctx[beam_indices], path_mask[beam_indices])

            log_probs = F.log_softmax(logit, dim=1).data # num x dim
            _, word_indices = logit.data.topk(min(beam_size, logit.size()[1]), dim=1) # num x beam_size
            word_scores = log_probs.gather(1, word_indices)
            assert word_scores.size() == word_indices.size()

            start_index = 0
            new_beams = []
            all_successors = []
            for beam_index, beam in enumerate(beams):
                successors = []
                end_index = start_index + len(beam)
                if beam:
                    for inf_index, (inf_state, word_score_row, word_index_row) in \
                        enumerate(zip(beam, word_scores[start_index:end_index], word_indices[start_index:end_index])):
                        for word_score, word_index in zip(word_score_row, word_index_row):
                            flat_index = start_index + inf_index
                            successors.append(
                                InferenceState(
                                    prev_inference_state=inf_state,
                                    flat_index=flat_index,
                                    last_word=word_index.item(),
                                    word_count=inf_state.word_count + 1,
                                    score=inf_state.score + word_score.item())
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_word == self.tok.word_to_index['<EOS>'] or t == self.args.maxDecode - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            if not any(beam for beam in beams):
                break

        words_batch = {}
        # max_len = 0
        for i in range(batch_size):
            path_id = obs[i]['path_id']
            if not path_id in words_batch:
                words_batch[path_id] = []
            this_completed = completed[i]
            this_completed = sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]
            for inf_state in this_completed:
                word_indices = backchain_inference_states(inf_state)
                words_batch[path_id].append(word_indices)
                # max_len = max(max_len,len(word_indices))
            
        # res = np.ones([batch_size, max_len]).astype(np.int32) * self.tok.word_to_index['<PAD>']
        # for i,words in enumerate(words_batch):
        #     for j,w in enumerate(words):
        #         res[i,j] = w

        return words_batch


    def get_insts_coco_beam(self, iters=None, wrapper=(lambda x: x)):
        '''
            return: 
                   path2inst: a dict {ID: [{ 'caption' : str}, ...], ...}
        '''
        # Get the caption for all the data

        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        if iters is None:
            total = self.env.size()
        else:
            total = iters
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            with torch.no_grad():
                words_batch = self.beam_search()  # Get the insts of the result
            # path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            # for path_id, inst in zip(path_ids, insts):
            #     path_id = str(path_id)
            #     if path_id not in path2inst:
            #         path2inst[path_id] = []

            #     inst = self.tok.decode_sentence(self.tok.shrink(inst))  # Shrink the words and decode
            #     path2inst[path_id].append({'caption': inst})
            for path_id in words_batch.keys():
                if path_id not in path2inst:
                    path2inst[str(path_id)] = []
                
                path2inst[str(path_id)] += [{'caption': self.tok.decode_sentence(self.tok.shrink(inst))} for inst in words_batch[path_id]]

        return path2inst

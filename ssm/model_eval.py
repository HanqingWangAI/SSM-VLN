import torch

import os
import time
import json
import numpy as np
from collections import defaultdict


from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch
from eval import Evaluation
from param import args

import torch.multiprocessing as mp
from learner import Learner

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")


def test():
    print('current directory',os.getcwd())
    os.chdir('..')
    print('current directory',os.getcwd())

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    visible_gpu = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu


    args.name = 'SSM'
    args.attn = 'soft'
    args.train = 'listener'
    args.featdropout = 0.3
    args.angle_feat_size = 128
    args.feedback = 'sample'
    args.ml_weight = 0.2
    args.sub_out = 'max'
    args.dropout = 0.5
    args.optim = 'adam'
    args.lr = 3e-4
    args.iters = 80000
    args.maxAction = 15
    args.batchSize = 4
    args.target_batch_size=4
    args.pe_dim = 128

    args.self_train = True
    args.aug = 'tasks/R2R/data/aug_paths.json'

    args.featdropout = 0.4
    args.iters = 200000

    if args.optim == 'rms':
        print("Optimizer: Using RMSProp")
        args.optimizer = torch.optim.RMSprop
    elif args.optim == 'adam':
        print("Optimizer: Using Adam")
        args.optimizer = torch.optim.Adam
    elif args.optim == 'sgd':
        print("Optimizer: sgd")
        args.optimizer = torch.optim.SGD



    TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
    TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

    IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'

    if args.features == 'imagenet':
        features = IMAGENET_FEATURES

    if args.fast_train:
        name, ext = os.path.splitext(features)
        features = name + "-fast" + ext


    print(args)


    def setup():
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        # Check for vocabs
        if not os.path.exists(TRAIN_VOCAB):
            write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
        if not os.path.exists(TRAINVAL_VOCAB):
            write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)
    #
    setup()

    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)

    print('start extract keys...')
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    print('keys extracted...')


    val_envs = {split: R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                             tokenizer=tok)
            for split in ['train', 'val_seen', 'val_unseen']}

    evaluators = {split: Evaluation([split], featurized_scans, tok)
            for split in ['train', 'val_seen', 'val_unseen']}
    

    learner = Learner(val_envs, "", tok, args.maxAction, process_num=4, max_node=17, visible_gpu=visible_gpu)
    learner.eval_init()
    
    ckpt = 'snap/%s/state_dict/ssm_ckpt' % args.name
        
    learner.load_eval(ckpt)

    results = learner.eval()
    loss_str = ''
    for key in results:
        evaluator = evaluators[key]
        result = results[key]

        score_summary, score_details = evaluator.score(result)

        loss_str += ", %s \n" % key
    
        for metric,val in score_summary.items():
            loss_str += ', %s: %.3f' % (metric, val)
            
        loss_str += '\n'


    print(loss_str)
    

if __name__ == "__main__":
    mp.set_start_method('spawn')
    test()
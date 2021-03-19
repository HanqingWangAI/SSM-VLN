import torch

import os
from collections import defaultdict


from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features, read_graph_features
import utils
from env import R2RBatch
from eval import Evaluation
from param import args

import torch.multiprocessing as mp
from learner import Learner

import warnings
warnings.filterwarnings("ignore")


def train():
    print('current directory',os.getcwd())
    os.chdir('..')
    print('current directory',os.getcwd())

    visible_gpu = "0,1,2,3" # avaiable GPUs, GPU0 is for processing gradient accumulating
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu


    args.name = 'SSM'
    args.attn = 'soft'
    args.train = 'listener'
    args.featdropout = 0.4
    args.angle_feat_size = 128
    args.feedback = 'sample'
    args.ml_weight = 0.2
    args.sub_out = 'max'
    args.dropout = 0.5
    args.optim = 'rms'
    args.lr = 1e-4
    args.iters = 80000
    args.maxAction = 15
    args.batchSize = 16
    args.aug = 'tasks/R2R/data/aug_paths.json'
    args.self_train = True
    

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



    log_dir = 'snap/%s' % args.name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                        splits=['train'], tokenizer=tok)
    aug_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                        splits=[args.aug], tokenizer=tok)

    train_env = {'train': train_env,'aug': aug_env}

    
    load_path = None
    
    torch.autograd.set_detect_anomaly(True)


    learner = Learner(train_env, "", tok, args.maxAction, process_num=4, max_node=17, visible_gpu=visible_gpu)

  
    if load_path is not None:
        print('load checkpoint from:', load_path)
        learner.load(load_path)


    learner.train()
    

if __name__ == "__main__":
    mp.set_start_method('spawn')
    train()
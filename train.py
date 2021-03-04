import copy
import numpy as np
import torch
import argparse
import logging
import time
import pdb
import os
import json
import random
import pickle
from utils import (
    evaluate
)
from tqdm import tqdm
from src.biosyn import (
    QueryDataset, 
    CandidateDataset, 
    DictionaryDataset,
    TextPreprocess, 
    RerankNet, 
    BioSyn
)

from IPython import embed

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Biosyn train')

    # Required
    parser.add_argument('--model_dir', required=True,
                        help='Directory for pretrained model')
    parser.add_argument('--train_dictionary_path', type=str, required=True,
                    help='train dictionary path')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--normalize_vecs',  action="store_true")
    parser.add_argument('--topk',  type=int, 
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    parser.add_argument('--initial_sparse_weight',
                        default=0, type=float)
    parser.add_argument('--dense_ratio', type=float,
                        default=0.5)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    
    parser.add_argument('--save_embeds', action="store_true")

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )

    return dictionary.data[:, 0::2]
    
def load_queries(data_dir, filter_composite, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def train(args, data_loader, model, **kwargs):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.train()
    
    if args.save_embeds:
        embeds_dir = os.path.join(args.output_dir, "embeds_{}".format(kwargs['epoch']))
    
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        
        if args.save_embeds:
            batch_x, batch_y, query_idx, batch_topk_idxs = data
        else:
            batch_x, batch_y = data

        batch_pred = model(batch_x)
        loss = model.get_loss(batch_pred, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        
        # Save the embeddings for the current step
        if args.save_embeds:
            LOGGER.info("Epoch {}/{} Step {}/{} embeddings serialization".format(kwargs['epoch'], args.epoch, train_steps, len(data_loader)))

            # Dense embeddings of the training queries
            train_query_dense_embeds = kwargs['biosyn'].embed_dense(
                names=kwargs['names_in_train_queries'], show_progress=True)

            # Dense embeddings of the training dictionary
            train_dict_dense_embeds = kwargs['biosyn'].embed_dense(
                names=kwargs['names_in_train_dictionary'], show_progress=True)
            
            # get dense nearest neighbors
            train_dense_candidate_idxs, _  = kwargs['biosyn'].get_dense_knn(
                train_query_dense_embeds,
                train_dict_dense_embeds,
                args.topk)
            
            # Save to the given path
            np.save(os.path.join(embeds_dir, str(train_steps) + '.npy'), train_dict_dense_embeds)
            np.save(os.path.join(embeds_dir, str(train_steps) + '_query_embeds.npy'), train_query_dense_embeds)
            np.save(os.path.join(embeds_dir, str(train_steps) + '_topk.npy'), batch_topk_idxs)
            np.save(os.path.join(embeds_dir, str(train_steps) + '_topk_by_queries.npy'), train_dense_candidate_idxs)
            np.save(os.path.join(embeds_dir, str(train_steps) + '_query_idx.npy'), query_idx)

        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss
    
def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # load dictionary and queries
    train_dictionary = load_dictionary(dictionary_path=args.train_dictionary_path)
    train_queries = load_queries(
        data_dir = args.train_dir, 
        filter_composite=True,
        filter_duplicate=True
    )

    # filter only names
    names_in_train_dictionary = train_dictionary[:,0]
    names_in_train_queries = train_queries[:,0]

    # load BERT tokenizer, dense_encoder, sparse_encoder
    biosyn = BioSyn()
    encoder, tokenizer = biosyn.load_bert(
        path=args.model_dir, 
        max_length=args.max_length,
        normalize_vecs=args.normalize_vecs,
        use_cuda=args.use_cuda,
    )
    sparse_encoder = biosyn.train_sparse_encoder(corpus=names_in_train_dictionary)
    sparse_weight = biosyn.init_sparse_weight(
        initial_sparse_weight=args.initial_sparse_weight,
        use_cuda=args.use_cuda
    )
    
    # load rerank model
    model = RerankNet(
        encoder=encoder,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        sparse_weight=sparse_weight,
        use_cuda=args.use_cuda
    )
    
    # embed sparse representations for query and dictionary
    # Important! This is one time process because sparse represenation never changes.
    LOGGER.info("Sparse embedding")
    
    train_query_sparse_embeds = biosyn.embed_sparse(
        names=names_in_train_queries, show_progress=True
    )
    train_dict_sparse_embeds = biosyn.embed_sparse(
        names=names_in_train_dictionary, show_progress=True
    )

    # get sparse knn
    sparse_knn = biosyn.get_sparse_knn(
        train_query_sparse_embeds,
        train_dict_sparse_embeds,
        args.topk
    )
    train_sparse_candidate_idxs, _ = sparse_knn

    # prepare for data loader of train and dev
    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = tokenizer, 
        topk = args.topk, 
        d_ratio=args.dense_ratio,
        s_query_embeds=train_query_sparse_embeds,
        s_dict_embeds=train_dict_sparse_embeds,
        s_candidate_idxs=train_sparse_candidate_idxs,
        return_idxs=args.save_embeds # indexes of top-ks are needed for save_embeds
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    start = time.time()
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")

        train_query_dense_embeds = biosyn.embed_dense(
            names=names_in_train_queries, show_progress=True
        )
        train_dict_dense_embeds = biosyn.embed_dense(
            names=names_in_train_dictionary, show_progress=True
        )
        
        # get dense knn
        dense_knn = biosyn.get_dense_knn(
            train_query_dense_embeds,
            train_dict_dense_embeds,
            args.topk
        )

        train_dense_candidate_idxs, _ = dense_knn

        if args.save_embeds:
            # Save the initially received dense embeddings
            LOGGER.info("Epoch {}/{} initial embeddings serialization".format(epoch, args.epoch))
            
            embeds_dir = os.path.join(args.output_dir, "embeds_{}".format(epoch))

            os.makedirs(embeds_dir, exist_ok=True)

            np.save(os.path.join(embeds_dir, 'initial.npy'), train_dict_dense_embeds)
            np.save(os.path.join(embeds_dir, 'initial_query_embeds.npy'), train_dict_dense_embeds)
            np.save(os.path.join(embeds_dir, 'initial_topk_by_queries.npy'), train_dense_candidate_idxs)

        # replace dense candidates in the train_set
        train_set.set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)

        # train
        # Save dense embeddings for the current epoch into npy file
        train_loss = train(
            args, data_loader=train_loader, model=model,
            epoch=epoch,
            biosyn=biosyn,
            names_in_train_queries=names_in_train_queries,
            names_in_train_dictionary=names_in_train_dictionary,
        )

        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        
        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            biosyn.save_model(checkpoint_dir)
        
        # save model last epoch
        if epoch == args.epoch:
            biosyn.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

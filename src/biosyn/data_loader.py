import re
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import logging
import pickle
import time
from tqdm import tqdm, trange

from IPython import embed

LOGGER = logging.getLogger(__name__)


class QueryDataset(Dataset):

    def __init__(self, data_dir, 
                filter_composite=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, filter_composite, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_composite, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        for concept_file in tqdm(concept_files):
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            for concept in concepts:
                concept = concept.split("||")
                pmid = concept[0]
                start_char, end_char = concept[1].split("|")
                mention = concept[3].strip()
                cui = concept[4].strip()
                sty = concept[2].strip()
                is_composite = (cui.replace("+","|").count("|") > 0)

                if filter_composite and is_composite:
                    continue
                else:
                    data.append((mention,cui,sty,pmid,start_char,end_char))
        
        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data


class DictionaryDataset():
    """
    A class used to load dictionary data
    """
    def __init__(self, dictionary_path):
        """
        Parameters
        ----------
        dictionary_path : str
            The path of the dictionary
        draft : bool
            use only small subset
        """
        LOGGER.info("DictionaryDataset! dictionary_path={}".format(
            dictionary_path 
        ))
        self.data = self.load_data(dictionary_path)
        
    def load_data(self, dictionary_path):
        name_cui_map = {}
        data = []
        with open(dictionary_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "": continue
                line_split = line.split("||")
                if len(line_split) == 2:
                    cui, name = line_split
                    sty = None
                elif len(line_split) == 3:
                    cui, sty, name = line_split
                else:
                    raise NotImplementedError()
                assert sty != ''
                data.append((name, sty, cui))
        
        data = np.array(data)
        return data


class CandidateDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self,
                 queries,
                 dicts,
                 tokenizer,
                 topk,
                 d_ratio,
                 s_query_embeds,
                 s_dict_embeds,
                 s_candidate_idxs):
        """
        Retrieve top-k candidates based on sparse/dense embedding

        Parameters
        ----------
        queries : list
            A list of tuples (name, id)
        dicts : list
            A list of tuples (name, id)
        tokenizer : BertTokenizer
            A BERT tokenizer for dense embedding
        topk : int
            The number of candidates
        d_ratio : float
            The ratio of dense candidates from top-k
        s_query_embeds : scipy.sparse.csr_matrix
            The sparse embeddings of all of the queries
        s_dict_embeds : scipy.sparse.csr_matrix
            The sparse embeddings of all of the dict items
        s_candidate_idxs : np.array
        """
        LOGGER.info("CandidateDataset! len(queries)={} len(dicts)={} topk={} d_ratio={}".format(
            len(queries),len(dicts), topk, d_ratio))
        self.query_names, self.query_ids = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_names, self.dict_ids = [row[0] for row in dicts], [row[1] for row in dicts]
        self.dict_ids_ndarray = np.array(self.dict_ids)
        self.topk = topk
        self.n_dense = int(topk * d_ratio)
        self.n_sparse = topk - self.n_dense
        self.tokenizer = tokenizer

        self.s_query_embeds = s_query_embeds
        self.s_dict_embeds = s_dict_embeds
        self.s_candidate_idxs = s_candidate_idxs
        self.d_candidate_idxs = None
        self.examples = {}

        #tokenized_names_file = 'tokenized_names.medmentions.pkl'
        #if not os.path.exists(tokenized_names_file):
        #    LOGGER.info("tokenizing query names!")
        #    self.all_query_tokens = {i : self.tokenizer.transform([query_name])
        #        for i, query_name in enumerate(tqdm(self.query_names))
        #    }

        #    LOGGER.info("tokenizing dict names!")
        #    self.all_dict_tokens = {i : self.tokenizer.transform([dict_name])
        #        for i, dict_name in enumerate(tqdm(self.dict_names))
        #    }
        #    with open(tokenized_names_file, 'wb') as f:
        #        pickle.dump(
        #            (self.all_query_tokens, self.all_dict_tokens), f
        #        )
        #else:
        #    with open(tokenized_names_file, 'rb') as f:
        #        _loaded_pkl = pickle.load(f)
        #    self.all_query_tokens, self.all_dict_tokens = _loaded_pkl
        LOGGER.info("tokenizing query names!")
        self.all_query_tokens = {i : self.tokenizer.transform([query_name])
            for i, query_name in enumerate(tqdm(self.query_names))
        }

        LOGGER.info("tokenizing dict names!")
        self.all_dict_tokens = {i : self.tokenizer.transform([dict_name])
            for i, dict_name in enumerate(tqdm(self.dict_names))
        }

    def set_dense_candidate_idxs(self, d_candidate_idxs):
        self.d_candidate_idxs = d_candidate_idxs
        self.rebuild_examples()

    def rebuild_examples(self):
        assert (self.s_candidate_idxs is not None)
        assert (self.s_query_embeds is not None)
        assert (self.s_dict_embeds is not None)
        assert (self.d_candidate_idxs is not None)

        LOGGER.info("Rebuilding candidate examples for dataloader!")
        for query_idx in trange(len(self.query_names)):
            # get the query token ids
            query_token = self.all_query_tokens[query_idx]

            # combine sparse and dense candidates as many as top-k
            s_candidate_idx = self.s_candidate_idxs[query_idx]
            d_candidate_idx = self.d_candidate_idxs[query_idx]

            # fill with sparse candidates first
            topk_candidate_idx = s_candidate_idx[:self.n_sparse]

            # fill remaining candidates with dense
            n_dense = self.topk - self.n_sparse
            novel_mask = ~np.in1d(d_candidate_idx, topk_candidate_idx)
            topk_candidate_idx = np.hstack(
                (topk_candidate_idx, d_candidate_idx[novel_mask][:n_dense])
            )
            
            # sanity check
            assert len(topk_candidate_idx) == self.topk
            assert len(topk_candidate_idx) == len(set(topk_candidate_idx))

            # compute the sparse scores from the embeddings
            s_query_vec = self.s_query_embeds[query_idx, :]
            s_dict_vec = self.s_dict_embeds[topk_candidate_idx, :]
            candidate_s_scores = np.dot(s_query_vec, s_dict_vec.T).todense()[0]
            candidate_s_scores = np.asarray(candidate_s_scores)

            labels = self.get_labels(query_idx, topk_candidate_idx).astype(np.float32)
            query_token = np.array(query_token).squeeze()
            candidate_tokens = [self.all_dict_tokens[cand_idx][0] 
                                    for cand_idx in topk_candidate_idx]
            candidate_tokens = np.asarray(candidate_tokens)

            self.examples[query_idx] = (
                (query_token, candidate_tokens, candidate_s_scores), labels
            )
    
    def __getitem__(self, query_idx):
        return self.examples[query_idx]

    def __len__(self):
        return len(self.query_names)

    def check_label(self, query_id, candidate_id_set):
        label = 0
        query_ids = query_id.split("|")
        """
        All query ids should be included in dictionary id
        """
        for q_id in query_ids:
            if q_id in candidate_id_set:
                label = 1
                continue
            else:
                label = 0
                break
        return label

    def get_labels(self, query_idx, candidate_idxs):
        labels = []
        query_id = self.query_ids[query_idx]
        candidate_ids = self.dict_ids_ndarray[candidate_idxs]
        for candidate_id in candidate_ids:
            label = self.check_label(query_id, candidate_id)
            labels.append(label)
        labels = np.asarray(labels)
        return labels

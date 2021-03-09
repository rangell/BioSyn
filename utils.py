import csv
import json
import math
import numpy as np
import pdb
from tqdm import tqdm
import faiss
import nmslib

from IPython import embed


def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)


def predict_topk(biosyn,
                 eval_dictionary,
                 eval_queries,
                 topk,
                 score_mode='hybrid',
                 type_given=False):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    encoder = biosyn.get_dense_encoder()
    tokenizer = biosyn.get_dense_tokenizer()
    sparse_encoder = biosyn.get_sparse_encoder()
    sparse_weight = biosyn.get_sparse_weight().item() # must be scalar value

    # useful if we're conditioning on types
    all_indv_types = [x for t in eval_dictionary[:,1] for x in t.split('|')]
    unique_types = np.unique(all_indv_types).tolist()
    v_check_type = np.vectorize(check_label)
    inv_idx = {t : v_check_type(eval_dictionary[:,1], t).nonzero()[0] 
                    for t in unique_types}

    # embed dictionary
    dict_sparse_embeds = biosyn.embed_sparse(names=eval_dictionary[:,0], show_progress=True)
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary[:,0], show_progress=True)

    # build the sparse index
    if not type_given:
        sparse_index = nmslib.init(
            method='hnsw',
            space='negdotprod_sparse_fast',
            data_type=nmslib.DataType.SPARSE_VECTOR
        )
        sparse_index.addDataPointBatch(dict_sparse_embeds)
        sparse_index.createIndex({'post': 2}, print_progress=False)
    else:
        sparse_index = {}
        for sty, indices in inv_idx.items():
            sparse_index[sty] = nmslib.init(
                method='hnsw',
                space='negdotprod_sparse_fast',
                data_type=nmslib.DataType.SPARSE_VECTOR
            )
            sparse_index[sty].addDataPointBatch(dict_sparse_embeds[indices])
            sparse_index[sty].createIndex({'post': 2}, print_progress=False)

    # build the dense index
    d = dict_dense_embeds.shape[1]
    if not type_given:
        nembeds = dict_dense_embeds.shape[0]
        if nembeds < 10000: # if the number of embeddings is small, don't approximate
            dense_index = faiss.IndexFlatIP(d)
            dense_index.add(dict_dense_embeds)
        else:
            nlist = int(math.floor(math.sqrt(nembeds))) # number of quantized cells
            nprobe = int(math.floor(math.sqrt(nlist))) # number of the quantized cells to probe
            quantizer = faiss.IndexFlatIP(d)
            dense_index = faiss.IndexIVFFlat(
                quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
            )
            dense_index.train(dict_dense_embeds)
            dense_index.add(dict_dense_embeds)
            dense_index.nprobe = nprobe
    else:
        dense_index = {}
        for sty, indices in inv_idx.items():
            sty_dict_dense_embeds = dict_dense_embeds[indices]
            nembeds = sty_dict_dense_embeds.shape[0]
            if nembeds < 10000: # if the number of embeddings is small, don't approximate
                dense_index[sty] = faiss.IndexFlatIP(d)
                dense_index[sty].add(sty_dict_dense_embeds)
            else:
                nlist = int(math.floor(math.sqrt(nembeds))) # number of quantized cells
                nprobe = int(math.floor(math.sqrt(nlist))) # number of the quantized cells to probe
                quantizer = faiss.IndexFlatIP(d)
                dense_index[sty] = faiss.IndexIVFFlat(
                    quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
                )
                dense_index[sty].train(sty_dict_dense_embeds)
                dense_index[sty].add(sty_dict_dense_embeds)
                dense_index[sty].nprobe = nprobe

    # respond to mention queries
    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        golden_cui = eval_query[1].replace("+","|")
        golden_sty = eval_query[2].replace("+","|")
        pmid = eval_query[3]
        start_char = eval_query[4]
        end_char = eval_query[5]

        dict_mentions = []
        for mention in mentions:

            mention_sparse_embeds = biosyn.embed_sparse(names=np.array([mention]))
            mention_dense_embeds = biosyn.embed_dense(names=np.array([mention]))

            # search the sparse index
            if not type_given:
                sparse_nn = sparse_index.knnQueryBatch(
                    mention_sparse_embeds, k=topk, num_threads=20
                )
            else:
                sparse_nn = sparse_index[golden_sty].knnQueryBatch(
                    mention_sparse_embeds, k=topk, num_threads=20
                )
            sparse_idxs, _ = zip(*sparse_nn)
            s_candidate_idxs = np.asarray(sparse_idxs)
            if type_given:
                # reverse mask index mapping
                s_candidate_idxs = inv_idx[golden_sty][s_candidate_idxs]
            s_candidate_idxs = s_candidate_idxs.astype(np.int64)

            # search the dense index
            if not type_given:
                _, d_candidate_idxs = dense_index.search(
                    mention_dense_embeds, topk
                )
            else:
                _, d_candidate_idxs = dense_index[golden_sty].search(
                    mention_dense_embeds, topk
                )
                # reverse mask index mapping
                d_candidate_idxs = inv_idx[golden_sty][d_candidate_idxs]
            d_candidate_idxs = d_candidate_idxs.astype(np.int64)

            # get the reduced candidate set
            reduced_candidate_idxs = np.unique(
                np.hstack(
                    [s_candidate_idxs.reshape(-1,),
                     d_candidate_idxs.reshape(-1,)]
                )
            )

            # get score matrix
            sparse_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_sparse_embeds, 
                dict_embeds=dict_sparse_embeds[reduced_candidate_idxs, :]
            ).todense()
            dense_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_dense_embeds, 
                dict_embeds=dict_dense_embeds[reduced_candidate_idxs, :]
            )

            if score_mode == 'hybrid':
                score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            elif score_mode == 'dense':
                score_matrix = dense_score_matrix
            elif score_mode == 'sparse':
                score_matrix = sparse_score_matrix
            else:
                raise NotImplementedError()

            # take care of getting the best indices
            candidate_idxs = biosyn.retrieve_candidate(
                score_matrix=score_matrix, 
                topk=topk
            )
            candidate_idxs = reduced_candidate_idxs[candidate_idxs]

            np_candidates = eval_dictionary[candidate_idxs].squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name':np_candidate[0],
                    'sty':np_candidate[1],
                    'cui':np_candidate[2],
                    'label':check_label(np_candidate[2], golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'pmid':pmid,
                'start_char':start_char,
                'end_char':end_char,
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result


def embed_and_index(biosyn, 
                    names):
    """
    Parameters
    ----------
    TODO: Add argument details
    """
    # Embed dictionary
    sparse_embeds = biosyn.embed_sparse(names=names, show_progress=True)
    dense_embeds = biosyn.embed_dense(names=names, show_progress=True)

    # Build sparse index
    sparse_index = nmslib.init(
        method='hnsw',
        space='negdotprod_sparse_fast',
        data_type=nmslib.DataType.SPARSE_VECTOR
    )
    sparse_index.addDataPointBatch(sparse_embeds)
    sparse_index.createIndex({'post': 2}, print_progress=False)

    # Build dense index
    d = dense_embeds.shape[1]
    nembeds = dense_embeds.shape[0]
    if nembeds < 10000:  # if the number of embeddings is small, don't approximate
        dense_index = faiss.IndexFlatIP(d)
        dense_index.add(dense_embeds)
    else:
        # number of quantized cells
        nlist = int(math.floor(math.sqrt(nembeds)))
        # number of the quantized cells to probe
        nprobe = int(math.floor(math.sqrt(nlist)))
        quantizer = faiss.IndexFlatIP(d)
        dense_index = faiss.IndexIVFFlat(
            quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
        )
        dense_index.train(dense_embeds)
        dense_index.add(dense_embeds)
        dense_index.nprobe = nprobe

    # Return embeddings and indexes
    return sparse_embeds, dense_embeds, sparse_index, dense_index


def get_query_nn(biosyn,
                 topk, 
                 sparse_embeds, 
                 dense_embeds, 
                 sparse_index, 
                 dense_index, 
                 q_sparse_embed, 
                 q_dense_embed):
    """
    Parameters
    ----------
    TODO: Add argument details
    """
    # Get sparse similarity weight to final score
    score_sparse_wt = biosyn.get_sparse_weight().item()

    # Find sparse-index k nearest neighbours
    sparse_knn = sparse_index.knnQueryBatch(
        q_sparse_embed, k=topk, num_threads=20)
    sparse_knn_idxs, _ = zip(*sparse_knn)
    sparse_knn_idxs = np.asarray(sparse_knn_idxs).astype(np.int64)
    # Find dense-index k nearest neighbours
    _, dense_knn_idxs = dense_index.search(q_dense_embed, topk)
    dense_knn_idxs = dense_knn_idxs.astype(np.int64)

    # Get unique candidates
    cand_idxs = np.unique(np.concatenate(
        sparse_knn_idxs.flatten(), dense_knn_idxs.flatten()))

    # Compute query-candidate similarity scores
    sparse_scores = biosyn.get_score_matrix(
        query_embeds=q_sparse_embed,
        dict_embeds=sparse_embeds[cand_idxs, :]
    ).todense().flatten()
    dense_scores = biosyn.get_score_matrix(
        query_embeds=q_sparse_embed,
        dict_embeds=dense_embeds[cand_idxs, :]
    ).flatten()
    if score_mode == 'hybrid':
        scores = score_sparse_wt * sparse_scores + dense_scores
    elif score_mode == 'dense':
        scores = dense_scores
    elif score_mode == 'sparse':
        scores = sparse_scores
    else:
        raise NotImplementedError()

    return cand_idxs, scores


def predict_topk_cluster_link(biosyn,
                              eval_dictionary,
                              eval_queries,
                              topk,
                              score_mode='hybrid'):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    
    Naming Convention
    -----------------
    - Dictionary == Entities
    - Queries == Mentions
    
    Assumptions
    -----------
    - Type is not given
    - Predictions must be returned for every query mention
    - No composites
    """
    n_entities = eval_dictionary.shape[0]
    n_mentions = eval_queries.shape[0]

    # Initialize a graph to store mention-mention and mention-entity similarity score edges
    joint_graph = {
        rows: np.array([]),
        cols: np.array([]),
        data: np.array([]),
        shape: (n_mentions, n_entities+n_mentions)
    }

    # Embed dictionary and build indexes
    dict_sparse_embeds, dict_dense_embeds, dict_sparse_index, dict_dense_index = embed_and_index(
        biosyn, eval_dictionary[:, 0])

    # Embed mention queries and build indexes
    men_sparse_embeds, men_dense_embeds, men_sparse_index, men_dense_index = embed_and_index(
        biosyn, eval_queries[:, 0])

    # Find topK similar entities and mentions for each mention query
    for eval_query_idx, eval_query in enumerate(tqdm(eval_queries, total=len(eval_queries))):
        # Slicing to get an array
        men_sparse_embed = men_sparse_embeds[eval_query_idx:eval_query_idx+1]
        men_dense_embed = men_dense_embeds[eval_query_idx:eval_query_idx+1]

        # Fetch nearest-neighbour entity candidates
        dict_cand_idxs, dict_cand_scores = get_query_nn(
            biosyn, topk, dict_sparse_embeds, dict_dense_embeds, dict_sparse_index, dict_dense_index, men_sparse_embed, men_dense_embed)
        # Add mention-entity edges to the joint graph
        joint_graph.rows = np.append(
            joint_graph.rows, [eval_query_idx]*len(dict_cand_idxs))
        joint_graph.cols = np.append(joint_graph.cols, dict_cand_idxs)
        joint_graph.data = np.append(joint_graph.data, dict_cand_scores)

        # Fetch nearest-neighbour mention candidates
        men_cand_idxs, men_cand_scores = get_query_nn(
            biosyn, topk+1, men_sparse_embeds, men_dense_embeds, men_sparse_index, men_dense_index, men_sparse_embed, men_dense_embed)
        # Filter returned candidates to remove the mention query
        men_cand_idxs, men_cand_scores = men_cand_idxs[np.where(
            men_cand_idxs != eval_query_idx)], men_cand_scores[np.where(men_cand_idxs != eval_query_idx)]
        # Add mention-mention edges to the joint graph
        joint_graph.rows = np.append(
            joint_graph.rows, [eval_query_idx]*len(men_cand_idxs))
        joint_graph.cols = np.append(
            joint_graph.cols, n_entities+men_cand_idxs)
        joint_graph.data = np.append(joint_graph.data, men_cand_scores)

        # Filter duplicates from graph
        joint_graph.rows, joint_graph.cols, joint_graph.data = zip(
            *set(zip(joint_graph.rows, joint_graph.cols, joint_graph.data)))
    return None


def evaluate(biosyn,
             eval_dictionary,
             eval_queries,
             topk,
             score_mode='hybrid',
             type_given=False):
    """
    predict topk and evaluate accuracy
    
    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    eval_dictionary : str
        dictionary to evaluate
    eval_queries : str
        queries to evaluate
    topk : int
        the number of topk predictions
    score_mode : str
        hybrid, dense, sparse
    type_given : bool
        whether or not to restrict entity set to ones with gold type

    Returns
    -------
    result : dict
        accuracy and candidates
    """
    result = predict_topk(
        biosyn, eval_dictionary, eval_queries, topk, score_mode, type_given
    )
    result = evaluate_topk_acc(result)
    
    return result

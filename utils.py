import csv
import json
import math
import numpy as np
import pdb
from tqdm import tqdm
import faiss
import nmslib
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from special_partition.special_partition import special_partition
from collections import defaultdict
import pickle

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
    biosyn : BioSyn
        trained biosyn model
    names : array
        list of names to embed and index

    Returns
    -------
    sparse_embeds : ndarray
        matrix of sparse embeddings
    dense_embeds : ndarray
        matrix of dense embeddings
    sparse_index : nmslib
        nmslib index of the sparse embeddings
    dense_index : faiss
        faiss index of the sparse embeddings
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
                 q_dense_embed,
                 score_mode):
    """
    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    topk : int
        the number of nearest-neighbour candidates to retrieve
    sparse_embeds : ndarray
        matrix of sparse embeddings
    dense_embeds : ndarray
        matrix of dense embeddings
    sparse_index : nmslib
        nmslib index of the sparse embeddings
    dense_index : faiss
        faiss index of the sparse embeddings
    q_sparse_embed : ndarray
        2-D array containing the sparse query embedding
    q_dense_embed : ndarray
        2-D array containing the dense query embedding
    score_mode : str
        "hybrid", "dense", "sparse"

    Returns
    -------
    cand_idxs : array
        nearest neighbour indices for the query, sorted in descending order of scores
    scores : array
        similarity scores for each nearest neighbour, sorted in descending order
    """
    # To accomodate the approximate-nature of the knn procedure, retrieve more samples and then filter down
    k = max(16, 2*topk)

    # Get sparse similarity weight to final score
    score_sparse_wt = biosyn.get_sparse_weight().item()

    # Find sparse-index k nearest neighbours
    sparse_knn = sparse_index.knnQueryBatch(
        q_sparse_embed, k=k, num_threads=20)
    sparse_knn_idxs, _ = zip(*sparse_knn)
    sparse_knn_idxs = np.asarray(sparse_knn_idxs).astype(np.int64)
    # Find dense-index k nearest neighbours
    _, dense_knn_idxs = dense_index.search(q_dense_embed, k)
    dense_knn_idxs = dense_knn_idxs.astype(np.int64)

    # Get unique candidates
    cand_idxs = np.unique(np.concatenate(
        (sparse_knn_idxs.flatten(), dense_knn_idxs.flatten())))

    # Compute query-candidate similarity scores
    sparse_scores = biosyn.get_score_matrix(
        query_embeds=q_sparse_embed,
        dict_embeds=sparse_embeds[cand_idxs, :]
    ).todense().getA1()
    dense_scores = biosyn.get_score_matrix(
        query_embeds=q_dense_embed,
        dict_embeds=dense_embeds[cand_idxs, :]
    ).flatten()
    if score_mode == 'hybrid':
        scores = score_sparse_wt * sparse_scores + dense_scores
    elif score_mode == 'dense':
        scores = dense_scores
    elif score_mode == 'sparse':
        scores = sparse_scores
    else:
        raise ValueError()

    # Return the topk neighbours
    
    cand_idxs, scores = zip(
        *sorted(zip(cand_idxs, scores), key=lambda x: -x[1]))
    return np.array(cand_idxs[:topk]), np.array(scores[:topk])


def partition_graph(graph, n_entities, return_clusters=False):
    """
    Parameters
    ----------
    graph : dict
        object containing rows, cols, data, and shape of the entity-mention joint graph
    n_entities : int
        number of entities in the dictionary
    return_clusters : bool
        flag to indicate if clusters need to be returned from the partition

    Returns
    -------
    partitioned_graph : coo_matrix
        partitioned graph with each mention connected to only one entity
    clusters : dict
        (optional) contains arrays of connected component indices of the graph
    """
    # Make the graph symmetric - needed for cluster inference after partitioning
    # _row = np.concatenate((graph['rows'], graph['cols']))
    # _col = np.concatenate((graph['cols'], graph['rows']))
    # _data = np.concatenate((graph['data'], graph['data']))
    
    # Filter duplicates
    seen = set()
    _f_row, _f_col, _f_data = [], [], []
    for k, _ in enumerate(_row):
        if (_row[k], _col[k]) in seen:
            continue
        seen.add((_row[k], _col[k]))
        _f_row.append(_row[k])
        _f_col.append(_col[k])
        _f_data.append(_data[k])
    _row, _col, _data = list(map(np.array, (_f_row, _f_col, _f_data)))

    # Sort data for efficient DFS
    tuples = zip(_row, _col, _data)
    tuples = sorted(tuples, key=lambda x: (x[1], -x[0]))
    special_row, special_col, special_data = zip(*tuples)
    special_row = np.asarray(special_row, dtype=np.int)
    special_col = np.asarray(special_col, dtype=np.int)
    special_data = np.asarray(special_data)

    # Construct the coo matrix
    graph = coo_matrix(
        (special_data, (special_row, special_col)),
        shape=graph['shape'])

    # Create siamese indices for simple lookup during partitioning
    edge_indices = {e: i for i, e in enumerate(zip(special_row, special_col))}
    siamese_indices = [edge_indices[(c, r)]
                       for r, c in zip(special_row, special_col)]
    siamese_indices = np.asarray(siamese_indices)

    # Order the edges in ascending order of similarity scores
    ordered_edge_indices = np.argsort(special_data)

    # Determine which edges to keep in the partitioned graph
    keep_edge_mask = special_partition(
        special_row,
        special_col,
        ordered_edge_indices,
        siamese_indices,
        n_entities)

    # Construct the partitioned graph
    partitioned_graph = coo_matrix(
        (special_data[keep_edge_mask],
        (special_row[keep_edge_mask], special_col[keep_edge_mask])),
        shape=graph.shape)
    
    if return_clusters:
        # Get an array with each graph index marked with the component label that it is connected to
        _, cc_labels = connected_components(
            csgraph=partitioned_graph,
            directed=False,
            return_labels=True)
        # Store clusters of indices marked with labels with at least 2 connected components
        unique_cc_labels, cc_sizes = np.unique(cc_labels, return_counts=True)
        filtered_labels = unique_cc_labels[cc_sizes > 1]
        clusters = defaultdict(list)
        for i, cc_label in enumerate(cc_labels):
            if cc_label in filtered_labels:
                clusters[cc_label].append(i)
        return partitioned_graph, clusters

    return partitioned_graph

def analyzeClusters(clusters, eval_dictionary, eval_queries, topk, debug_mode):
    """
    Parameters
    ----------
    clusters : dict
        contains arrays of connected component indices of a graph
    eval_dictionary : ndarray
        entity dictionary to evaluate
    eval_queries : ndarray
        mention queries to evaluate
    topk : int
        the number of nearest-neighbour mention candidates considered
    debug_mode : bool
        Flag to enable reporting debug statistics
    
    Returns
    -------
    results : dict
        Contains n_entities, n_mentions, k_candidates, accuracy, success[], failure[]
    """
    n_entities = eval_dictionary.shape[0]
    n_mentions = eval_queries.shape[0]

    results = {
        'n_entities': n_entities,
        'n_mentions': n_mentions,
        'k_candidates': topk,
        'accuracy': 0,
        'failure': [],
        'success': []
    }
    _debug_n_mens_evaluated, _debug_clusters_wo_entities, _debug_clusters_w_mult_entities = 0, 0, 0

    for cluster in tqdm(clusters.values()):
        # The lowest value in the cluster should always be the entity
        pred_entity_idx = cluster[0]
        # Track the graph index of the entity in the cluster
        pred_entity_idxs = [pred_entity_idx]
        if pred_entity_idx >= n_entities:
            # If the first element is a mention, then the cluster does not have an entity
            _debug_clusters_wo_entities += 1
            continue
        pred_entity = eval_dictionary[pred_entity_idx]
        pred_entity_cuis = pred_entity[2].replace('+', '|').split('|')
        _debug_tracked_mult_entities = False
        for i in range(1, len(cluster)):
            men_idx = cluster[i] - n_entities
            if men_idx < 0:
                # If elements after the first are entities, then the cluster has multiple entities
                if not _debug_tracked_mult_entities:
                    _debug_clusters_w_mult_entities += 1
                    _debug_tracked_mult_entities = True
                # Track the graph indices of each entity in the cluster
                pred_entity_idxs.append(cluster[i])
                # Predict based on all entities in the cluster
                pred_entity_cuis += list(set(eval_dictionary[cluster[i]][2].replace('+', '|').split('|')) - set(pred_entity_cuis))
                continue
            _debug_n_mens_evaluated += 1
            men_query = eval_queries[men_idx]
            men_golden_cuis = men_query[1].replace('+', '|').split('|')
            report_obj = {
                'pm_id': men_query[3],
                'mention_name': men_query[0],
                'mention_gold_cui': men_query[1],
                'predicted_name': '|'.join(eval_dictionary[pred_entity_idxs,0]),
                'predicted_cui': '|'.join(pred_entity_cuis),
            }
            if debug_mode:
                report_obj['graph_mention_idx'] = cluster[i]
                report_obj['graph_entity_idx'] = '|'.join(map(str, pred_entity_idxs))
            # Correct prediction
            if not set(pred_entity_cuis).isdisjoint(men_golden_cuis):
                results['accuracy'] += 1
                results['success'].append(report_obj)
            # Incorrect prediction
            else:
                results['failure'].append(report_obj)
    results['accuracy'] = f"{results['accuracy'] / float(_debug_n_mens_evaluated if debug_mode else n_mentions) * 100} %"

    if debug_mode:
        # Report debug statistics
        results['n_mentions_evaluated'] = _debug_n_mens_evaluated
        results['n_clusters'] = len(clusters)
        results['n_clusters_wo_entities'] = _debug_clusters_wo_entities
        results['n_clusters_w_mult_entities'] = _debug_clusters_w_mult_entities
    else:
        # Run sanity checks
        assert n_mentions == _debug_n_mens_evaluated
        assert _debug_clusters_wo_entities == 0
        assert _debug_clusters_w_mult_entities == 0

    return results


def predict_topk_cluster_link(biosyn,
                              eval_dictionary,
                              eval_queries,
                              topk,
                              output_dir,
                              score_mode='hybrid',
                              debug_mode=False):
    """
    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    eval_dictionary : ndarray
        entity dictionary to evaluate
    eval_queries : ndarray
        mention queries to evaluate
    topk : int
        the number of nearest-neighbour mention candidates to consider
    output_dir : str
        output directory path for intermediate files and results
    score_mode : str
        "hybrid", "dense", "sparse"
    debug_mode : bool
        Flag to enable reporting debug statistics
    
    Returns
    -------
    results : array
        Contains result dictionaries corresponding to each value of k, which each contain n_entities, n_mentions, k_candidates, accuracy, success[], failure[]
  
    Assumptions
    -----------
    - type is not given
    - no composites
    - predictions returned for every query mention
    """
    n_entities = eval_dictionary.shape[0]
    n_mentions = eval_queries.shape[0]

    # Values of k to run the evaluation against
    topk_vals = [0, *[2**i for i in range(int(math.log(topk, 2)) + 1)]]
    # Store the maximum evaluation k
    topk = topk_vals[-1]

    # Check if graphs are already built
    if __import__('os').path.isfile(f'{output_dir}/graphs.pickle'):
        with open(f'{output_dir}/graphs.pickle', 'rb') as read_handle:
            joint_graphs = pickle.load(read_handle)
    else:
        # Initialize graphs to store mention-mention and mention-entity similarity score edges;
        # Keyed on the k-nearest mentions retrieved
        joint_graphs = {}
        for k in topk_vals:
            joint_graphs[k] = {
                'rows': np.array([]),
                'cols': np.array([]),
                'data': np.array([]),
                'shape': (n_entities+n_mentions, n_entities+n_mentions)
            }

        # Embed entity dictionary and build indexes
        dict_sparse_embeds, dict_dense_embeds, dict_sparse_index, dict_dense_index = embed_and_index(
            biosyn, eval_dictionary[:, 0])

        # Embed mention queries and build indexes
        men_sparse_embeds, men_dense_embeds, men_sparse_index, men_dense_index = embed_and_index(
            biosyn, eval_queries[:, 0])

        # Find the most similar entity and topk mentions for each mention query
        for eval_query_idx, eval_query in enumerate(tqdm(eval_queries, total=len(eval_queries))):
            men_sparse_embed = men_sparse_embeds[eval_query_idx:eval_query_idx+1] # Slicing to get a 2-D array
            men_dense_embed = men_dense_embeds[eval_query_idx:eval_query_idx+1]

            # Fetch nearest entity candidate
            dict_cand_idx, dict_cand_score = get_query_nn(
                biosyn, 1, dict_sparse_embeds, dict_dense_embeds, dict_sparse_index, 
                dict_dense_index, men_sparse_embed, men_dense_embed, score_mode)

            # Fetch (k+1) NN mention candidates
            men_cand_idxs, men_cand_scores = get_query_nn(
                biosyn, topk + 1, men_sparse_embeds, men_dense_embeds, men_sparse_index,
                men_dense_index, men_sparse_embed, men_dense_embed, score_mode)
            # Filter candidates to remove mention query and keep only the top k candidates
            filter_mask = men_cand_idxs != eval_query_idx
            if not np.all(filter_mask):
                men_cand_idxs, men_cand_scores = men_cand_idxs[filter_mask], men_cand_scores[filter_mask]
            else:
                men_cand_idxs, men_cand_scores = men_cand_idxs[:topk], men_cand_scores[:topk]

            # Add edges to the graphs
            for k in joint_graphs:
                joint_graph = joint_graphs[k]
                # Add mention-entity edge
                joint_graph['rows'] = np.append(
                    joint_graph['rows'], [n_entities+eval_query_idx]) # Mentions added at an offset of maximum entities
                joint_graph['cols'] = np.append(joint_graph['cols'], dict_cand_idx)
                joint_graph['data'] = np.append(joint_graph['data'], dict_cand_score)
                if k > 0:
                    # Add mention-mention edges
                    joint_graph['rows'] = np.append(
                        joint_graph['rows'], [n_entities+eval_query_idx]*len(men_cand_idxs[:k]))
                    joint_graph['cols'] = np.append(
                        joint_graph['cols'], n_entities+men_cand_idxs[:k])
                    joint_graph['data'] = np.append(joint_graph['data'], men_cand_scores[:k])
        
        # Pickle the graphs
        with open(f'{output_dir}/graphs.pickle', 'wb') as write_handle:
            pickle.dump(joint_graphs, write_handle, protocol=pickle.HIGHEST_PROTOCOL)

    results = []
    for k in joint_graphs:
        # Partition graph based on cluster-linking constraints
        partitioned_graph, clusters = partition_graph(
            joint_graphs[k], n_entities, return_clusters=True)
        # Infer predictions from clusters
        result = analyzeClusters(clusters, eval_dictionary, eval_queries, k, debug_mode)
        # Store result
        results.append(result)
    return results


def evaluate(biosyn,
             eval_dictionary,
             eval_queries,
             topk,
             output_dir,
             score_mode='hybrid',
             type_given=False,
             use_cluster_linking=False,
             debug_mode=False):
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
    output_dir : str
        output directory path for intermediate files and results
    score_mode : str
        hybrid, dense, sparse
    type_given : bool
        whether or not to restrict entity set to ones with gold type
    use_cluster_linking : bool
        flag indicating whether the cluster linking inference should be applied or not
    debug_mode : bool
        Flag to enable reporting debug statistics for cluster linking

    Returns
    -------
    result : dict or array
        accuracy and candidates
    """
    if use_cluster_linking:
        result = predict_topk_cluster_link(
            biosyn, eval_dictionary, eval_queries, topk, output_dir, score_mode, debug_mode)
    else:
        result = predict_topk(
            biosyn, eval_dictionary, eval_queries, topk, score_mode, type_given)
        result = evaluate_topk_acc(result)

    return result

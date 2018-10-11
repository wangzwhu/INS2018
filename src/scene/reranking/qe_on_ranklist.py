import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())
from src import util

def alpha_query_expansion(query_features, gallery_features, ranklist, N=5, alpha=0):
    """
    Alpha Query Expansion function. It created a new query based on the top N
    retrieved results.
    args:
        ranklist: ranking list of gallery index for given queries.
        query_features: the global feature of query
        gallery_features: the global feature of gallery
        N :  Top results to generate new query (N=100)
        alpha : alpha parameter used in query expansion, when alpha is given as 0,
        this function is equivalent to average query expansion.
    """

    # get scores
    num_query_feartures = query_features.shape[0]
    new_queries = []
    for i in tqdm(range(num_query_feartures)):
        total_weight = 1
        total_feature = query_features[i,:]
        # get rankings
        idx = ranklist[i, :]
        # get features
        new_query_feature = query_features[i, :]
        for k in range(N):
            weight = np.power(new_query_feature.dot(gallery_features[idx[k], :]), alpha)
            total_feature = total_feature + np.multiply(weight, gallery_features[idx[k], :])
            total_weight = total_weight + weight
        new_query_feature = np.divide(total_feature, total_weight)
        new_queries.append(new_query_feature)
    return np.array(new_queries)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_feature_dir',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir-qe',
            help='Directory to query features.')
    parser.add_argument('--gallery_feature_dir',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir-dba',
            help='Directory to gallery features.')
    parser.add_argument('--ranklist_path',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir-dba-qe/index_full.npy',
            help='Path to the initial searched ranklist.')
    parser.add_argument('--index_path',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir-dba-qe/index_x1.npy',
            help='Path to the index results after qe.')
    parser.add_argument('--distance_path',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir-dba-qe/distance_x1.npy',
            help='Path to the score results after qe.')
    parser.add_argument('--gpu_ids',
            type=str,
            default='0,1',
            help='specify which gpu to use')
    args = parser.parse_args()
    # get features
    query_features = np.load(os.path.join(args.query_feature_dir, 'cache_features.npy'))
    print('query features: ', query_features.shape)
    gallery_features = util.load_features(args.gallery_feature_dir, 'gallery', '.jbl')
    print('gallery features: ', gallery_features.shape)
    # get ranklist
    ranklist = np.load(args.ranklist_path)
    print('ranklist: ', ranklist.shape)
    # query expansion
    print('start query expansion')
    new_queries = alpha_query_expansion(query_features, gallery_features, ranklist)
    # start to search
    print('start searching')
    knn = util.KNN(gallery_features, 'euclidean', gpu_ids=args.gpu_ids)
    dis, ind = knn.search(new_queries, gallery_features.shape[0])
    # save searched resutls
    np.save(args.index_path, ind)
    np.save(args.distance_path, dis)

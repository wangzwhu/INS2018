import numpy as np
import os
import glob
import joblib
import tqdm
import argparse
import more_itertools
import sys
sys.path.append(os.getcwd())
from src import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_feature_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir/query',
                        help="Directory to query features."
                        )
    parser.add_argument('--new_query_feature_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir-qe',
                        help="Directory to new query features."
                        )
    parser.add_argument('--database_feature_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir-dba',
                        help='Directory to database features.'
                        )
    parser.add_argument('--search_results_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir-dba-qe/',
                        help='Directory to initial search results.'
                        )
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='Specify which gpu to use for computation.'
                        )
    parser.add_argument('--topk',
                        type=int,
                        default=1000,
                        help='Top K results to retrieve.'
                        )

    args = parser.parse_args()
    query_features = util.new_expansion_featuers(args.query_feature_dir, args.new_query_feature_dir,
            cache_file=None, use_cache=True)
    database_features = util.load_features(args.database_feature_dir, 'database',
            '.jbl', cache_file=None, use_cache=True)
    print('database features: {}'.format(database_features.shape))

    # start to search
    knn = util.KNN(database_features, 'euclidean', args.gpu_ids)
    dis, ind = knn.search(query_features, database_features.shape[0])
    print('distance:', dis.shape, 'index:', ind.shape)
    np.save(os.path.join(args.search_results_dir, 'index_full.npy'), ind)
    np.save(os.path.join(args.search_results_dir, 'distance_full.npy'), dis)

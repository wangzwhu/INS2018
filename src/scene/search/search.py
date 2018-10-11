import numpy as np
import os
from tqdm import tqdm
import argparse
import more_itertools
import sys
sys.path.append(os.getcwd())
from src.scene import util
from src.scene.search.knn import KNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_feature_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir/query',
                        help="Directory to query features."
                        )
    parser.add_argument('--database_feature_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir/gallery',
                        help='Directory to database features.'
                        )
    parser.add_argument('--mapper_path',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir/mapper.npy',
                        help='Path to database features mapper.'
                        )
    parser.add_argument('--search_results_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir/',
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
    # load features
    query_features = util.load_features(args.query_feature_dir, 'query',
            '.jbl', cache_file=None, use_cache=True)
    print('query features: {}'.format(query_features))
    database_features = util.load_features(args.database_feature_dir, 'database',
            '.jbl', cache_file=None, use_cache=True)
    print('database features: {}'.format(database_features))

    # ompute a mapper from feature-num to image-num and write it.
    if os.path.exists(args.mapper_path):
        print('Mapper exsits. Loading mapper...')
        mapper = np.load(args.mapper_path)
    else:
        mapper = []
        for image_num, vecs in enumerate(tqdm(database_features)):
            mapper.extend([image_num for _ in range(vecs.shape[0])])
        np.save(args.mapper_path, np.array(mapper))
        print('saving mapper to {}'.format(args.mapper_path))

    # start to search
    K = 500000
    knn = KNN(database_features, 'euclidean', args.gpu_ids)
    dis, ind = knn.search(query_features, K)
    print(dis, ind)

    # As each video contains several frames, we need to map the ind from framce space to video space.
    index = []
    for row_num in ind.shape[0]:
        map_ind = [mapper[_] for _ in ind[row_num]]
        index.append(list(more_itertools.unique_everseen(map_ind))[:, :args.topk])
    print('index : {}'.format(index))
    print('index shape : {}'.format(index.shape))
    np.save(os.path.join(args.search_results_dir, 'index.npy'), index)

    round_distance = round(dis)
    distance = list(more_itertools.unique_everseen(round_distance))[:, :args.topk]
    print('distance : {}'.format(index))
    print('distance shape : {}'.format(distance.shape))
    np.save(os.path.join(args.search_results_dir, 'distance.npy'), distance)

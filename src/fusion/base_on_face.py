import numpy as np
import os
import tqdm
import joblib
import argparse
from tqdm import tqdm
import more_itertools
import sys
sys.path.append(os.getcwd())
from src import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_feature_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir-qe',
                        help="Directory to query features."
                        )
    parser.add_argument('--database_feature_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir-dba',
                        help='Directory to database features.'
                        )
    parser.add_argument('--shot_orders_path',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/clips.txt',
                        help='Path to the clips file which lists the path of all shots in an order.'
                        )
    parser.add_argument('--face_results_path',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/facenet/distance_full.npy',
                        help='Directory to initial search results.'
                        )
    parser.add_argument('--search_results_dir',
                        type=str,
                        default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/fusion/base-on-face',
                        help='Directory to initial search results.'
                        )
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='Specify which gpu to use for computation.'
                        )
    parser.add_argument('--topk_face',
                        type=int,
                        default=5000,
                        help='Search based on union of top K face results.'
                        )
    parser.add_argument('--topk',
                        type=int,
                        default=1000,
                        help='Top K results to retrieve.'
                        )

    args = parser.parse_args()
    # get face distance results
    face_distance = np.load(args.face_results_path)
    print('face distance:', face_distance)
    sort_face_distance = np.argsort(face_distance)[:, :args.topk_face]
    print('after sort:', sort_face_distance)
    with open(args.shot_orders_path, 'r') as f:
        shot_orders = [shot_name.strip() for shot_name in f.readlines()]
    print('shot orders:', len(shot_orders))

    # load face and id dictionay
    face_dict = util.get_face_dict()
    # load scene features and person results for 30 topics, search results
    person_query_order, scene_query_order = util.get_query_order()
    distance = []
    index = []
    for person, scene in zip(person_query_order, scene_query_order):
        print('processing ', person, 'and ', scene + '...')
        index_results = sort_face_distance[face_dict[person]]
        database_features = [joblib.load(os.path.join(args.database_feature_dir, shot_orders[i]+'.jbl'))['feats']
                for i in tqdm(index_results)]
        database_features = np.array(database_features)
        query_feature = joblib.load(os.path.join(args.query_feature_dir, scene+'.jbl'))['feats'].reshape(1, -1)
        print('database features:', database_features.shape)
        print('query feature:', query_feature.shape)
        # start to search
        knn = util.KNN(database_features, 'euclidean', args.gpu_ids)
        dis, ind = knn.search(query_feature, args.topk_face)
        distance.append(dis)
        index.append(ind)
        print('distance:', dis.shape, 'index:', ind.shape)
        np.save(os.path.join(args.search_results_dir, 'index_'+person+'_'+scene+'.npy'), ind)
        np.save(os.path.join(args.search_results_dir, 'distance_'+person+'_'+scene+'.npy'), dis)
    distance = np.vstack(distance)
    index = np.vstack(index)
    print('distance:',distance.shape,'index:',index.shape)
    np.save(os.path.join(args.search_results_dir, 'index.npy'), index)
    np.save(os.path.join(args.search_results_dir, 'distance.npy'), distance)

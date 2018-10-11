import os
import sys
import scipy.io as scio
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--distance_path',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/fusion/base-on-face/distance.npy',
            help='Path to the distance matrix.'
            )
    parser.add_argument(
            '--index_path',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/fusion/base-on-face/index.npy',
            help='Path to the index matrix.'
            )
    parser.add_argument(
            '--shot_meta_file',
            type=str,
            default='/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/meta/Index.mat',
            help='Path to shot meta file.'
            )
    parser.add_argument(
            '--show_results_dir',
            type=str,
            default='/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/result/tv2018/test2018',
            help='Directory that contains results to show on the web.'
            )
    parser.add_argument(
            '--run_id',
            type=str,
            default='fusion_base_on_face',
            help='Id of scene results obtained by dir.'
            )
    parser.add_argument(
            '--topk',
            type=int,
            default=1000,
            help='top k results to show on web'
            )
    parser.add_argument(
            '--num_topics',
            type=int,
            default=30,
            help='number of topics to query'
            )
    args = parser.parse_args()
    # load distance and index
    distance = np.load(args.distance_path)
    index = np.load(args.index_path)
    # load shot meta data
    mat_data = scio.loadmat(args.shot_meta_file)
    shot_index = mat_data['Index']
    # make directory if not exsit
    if not os.path.exists(os.path.join(args.show_results_dir, args.run_id)):
        os.makedirs(os.path.join(args.show_results_dir, args.run_id))
    # define topic id and default query id
    topic_id_list = [str(topic) for topic in np.arange(9219, 9219+args.num_topics)]
    query_id = 'shot1_1'

    print(args.num_topics)
    for i in range(args.num_topics):
        topic_id = topic_id_list[i]
        print('top id:', topic_id)
        if not os.path.exists(os.path.join(args.show_results_dir, args.run_id, topic_id)):
            os.makedirs(os.path.join(args.show_results_dir, args.run_id, topic_id))
        with open(os.path.join(args.show_results_dir, args.run_id, topic_id, 'TRECVID2013_11.res'),'w') as f:
            for j in range(args.topk):
                shot_id = 'shot'+str(shot_index[index[i,j], 0])+'_'+str(shot_index[index[i,j], 1])
                f.write(shot_id + ' #$# ' + query_id + ' #$# ' + str(1.0/distance[i,j]) + '\n')

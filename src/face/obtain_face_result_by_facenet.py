from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import scipy.io as scio
import sys
sys.path.append("./facenet-master/src/")

import os
import argparse
import numpy as np
import random
from time import sleep
from sklearn.metrics.pairwise import euclidean_distances


def main():

    result_file_path = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/result/tv2018/test2018'
    run_id = 'person_0522_facenet_dict100'

    result_face_distance_dir = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/face_distance_0522.npy'
    shot_meta_file = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/meta/Index.mat'
    mat_data = scio.loadmat(shot_meta_file)
    shot_index = mat_data['Index']
    face_distance = np.load(result_face_distance_dir)
    face_distance[face_distance == 0] = 2
    if not os.path.exists(os.path.join(result_file_path, run_id)):
        os.makedirs(os.path.join(result_file_path, run_id))

    topic_id_list = ['9222', '9234', '9228', '9240', '9243', '9219', '9246', '9225', '9231', '9237']
    query_id = 'shot1_1'

    for i in range(10):
        topic_id = topic_id_list[i]
        print(topic_id)
        if not os.path.exists(os.path.join(result_file_path, run_id, topic_id)):
            os.makedirs(os.path.join(result_file_path, run_id, topic_id))

        output = open(os.path.join(result_file_path, run_id, topic_id, 'TRECVID2013_11.res'), 'w')

        person_face_distance = face_distance[:, i]
        person_rank_result = np.argsort(person_face_distance)
        for j in range(12000, 13000):
            shot_id = 'shot' + str(shot_index[person_rank_result[j], 0]) + '_' + str(shot_index[person_rank_result[j], 1])
            shot_score = str(1.0 / person_face_distance[person_rank_result[j]])
            output.write(shot_id + ' #$# ' + query_id + ' #$# ' + shot_score + '\n')

        output.close()


if __name__ == '__main__':
    main()

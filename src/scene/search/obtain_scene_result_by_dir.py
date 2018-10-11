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
from sklearn import preprocessing

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main():

    person_result_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/face_distance_0522.npy'
    scene_result_file  = '/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir-dba/distance_full.npy'
    scene_result_index_file = '/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir-dba/index_full.npy'

    gallery_num = 471526
    scene_num = 10
    person_num = 10

    # scene
    scene_distance_original = np.load(scene_result_file)
    scene_distance_original_index = np.load(scene_result_index_file)
    for i in range(90):
        sort_index = np.argsort(scene_distance_original_index[i, :])
        scene_distance_original[i, :] = scene_distance_original[i, sort_index]

    scene_distance = np.zeros((gallery_num, scene_num), dtype=float)
    scene_distance[:, 0] = np.min(scene_distance_original[0:12, :], 0)   # 12
    scene_distance[:, 1] = np.min(scene_distance_original[12:24, :], 0)  # 12
    scene_distance[:, 2] = np.min(scene_distance_original[24:30, :], 0)  # 6
    scene_distance[:, 3] = np.min(scene_distance_original[30:36, :], 0)  # 6
    scene_distance[:, 4] = np.min(scene_distance_original[36:42, :], 0)  # 6
    scene_distance[:, 5] = np.min(scene_distance_original[42:54, :], 0)  # 12
    scene_distance[:, 6] = np.min(scene_distance_original[54:60, :], 0)  # 6
    scene_distance[:, 7] = np.min(scene_distance_original[60:66, :], 0)  # 6
    scene_distance[:, 8] = np.min(scene_distance_original[66:78, :], 0)  # 12
    scene_distance[:, 9] = np.min(scene_distance_original[78:90, :], 0)  # 12
    scene_similarity = np.exp(-scene_distance)
    min_max_scaler = preprocessing.MinMaxScaler()
    scene_similarity = min_max_scaler.fit_transform(scene_similarity)

    # person
    person_distance  = np.load(person_result_file)
    person_distance[person_distance == 0] = 2
    person_similarity = np.exp(person_distance)
    min_max_scaler = preprocessing.MinMaxScaler()
    person_similarity = min_max_scaler.fit_transform(person_similarity)
    print('score loaded')

    topic_start_id = 9219
    result_file_path = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/result/tv2018/test2018'
    shot_meta_file = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/meta/Index.mat'
    mat_data = scio.loadmat(shot_meta_file)
    shot_index = mat_data['Index']

    run_id = '0607_dir_result'
    query_id = 'shot1_1'
    for i in range(10):
        topic_id = topic_start_id + i
        temp_scene  = scene_similarity[:, i]
        final_similarity_knn = temp_scene
        final_similarity_result = np.argsort(-final_similarity_knn)

        if not os.path.exists(os.path.join(result_file_path, run_id, str(topic_id))):
            os.makedirs(os.path.join(result_file_path, run_id, str(topic_id)))
        output = open(os.path.join(result_file_path, run_id, str(topic_id), 'TRECVID2013_11.res'), 'w')

        for j in range(1000):
            shot_id = 'shot' + str(shot_index[final_similarity_result[j], 0]) + '_' + str(shot_index[final_similarity_result[j], 1])
            shot_score = str(final_similarity_knn[final_similarity_result[j]])
            output.write(shot_id + ' #$# ' + query_id + ' #$# ' + shot_score + '\n')
        output.close()



if __name__ == '__main__':
    main()

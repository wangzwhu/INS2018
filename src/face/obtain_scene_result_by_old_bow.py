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

    scene_result_file_path_2016 = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/meta/tv2016/test2016/lst_location_by_query.mat'
    scene_result_file_path_2017 = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/meta/tv2017/test2017/lst_location_by_query.mat'
    shot_meta_file = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/meta/Index.mat'
    result_scene_distance_dir = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/scene_similarity_0523.npy'
    result_file_path = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/result/tv2018/test2018'

    mat_data = scio.loadmat(shot_meta_file)
    shot_index = mat_data['Index']

    gallery_num = 471526
    query_num  = 60
    scene_result = np.zeros((gallery_num, query_num), dtype=float)

    cafe1 = [30, 33, 37, 40, 47, 50, 56]        # cafe1     1
    foyer = [3, 7, 12, 17, 28]                  # Foyer     3
    kitchen1 = [1, 6, 10, 15, 20, 26]              # Kitchen1  4
    kitchen2 = [32, 36, 42, 49, 53, 58]            # Kitchen2  5
    laun = [2, 11, 16, 21, 23, 27, 34, 38, 41, 44, 48, 51, 55]  # Laun 6
    lr1 = [4, 8, 13, 18, 24, 29]              # LR1 7
    lr2 = [31, 35, 45, 52, 57]                # LR2 8
    pub = [0, 5, 9, 14, 19, 22, 25]           # pub 9
    market = [39, 43, 46, 54, 59]                # market 10

    mat_data = scio.loadmat(scene_result_file_path_2016)
    scene_shots = mat_data['lstLocationShot']
    scene_shots_scores = mat_data['loc_scores']
    cell_num = len(scene_shots[0])
    for i in range(cell_num):
        shot_num = len(scene_shots[0][i])
        for j in range(shot_num):
            shot_id = scene_shots[0][i][j][0][0]
            score = scene_shots_scores[0][i][j][0]
            index_a = shot_id.find('shot')
            index_b = shot_id.find('_')
            shot_id_num_1 = shot_id[index_a+4:index_b]
            shot_id_num_2 = shot_id[index_b+1: ]
            shot_position = list(set(np.where(shot_index[:, 0] == int(shot_id_num_1))[0]).intersection(set(np.where(shot_index[:, 1] == int(shot_id_num_2))[0])))[0]
            scene_result[shot_position, i] = score
        print(i)

    mat_data = scio.loadmat(scene_result_file_path_2017)
    scene_shots = mat_data['lstLocationShot']
    scene_shots_scores = mat_data['loc_scores']
    cell_num = len(scene_shots[0])
    for i in range(cell_num):
        shot_num = len(scene_shots[0][i])
        for j in range(shot_num):
            shot_id = scene_shots[0][i][j][0][0]
            score = scene_shots_scores[0][i][j][0]
            index_a = shot_id.find('shot')
            index_b = shot_id.find('_')
            shot_id_num_1 = shot_id[index_a+4:index_b]
            shot_id_num_2 = shot_id[index_b+1: ]
            shot_position = list(set(np.where(shot_index[:, 0] == int(shot_id_num_1))[0]).intersection(set(np.where(shot_index[:, 1] == int(shot_id_num_2))[0])))[0]
            scene_result[shot_position, i+30] = score
        print(i+30)

    scene_sim_result = np.zeros((gallery_num, 10), dtype=float)
    scene_sim_result[:, 0] = np.mean(scene_result[:, cafe1], 1)
    scene_sim_result[:, 2] = np.mean(scene_result[:, foyer], 1)
    scene_sim_result[:, 3] = np.mean(scene_result[:, kitchen1], 1)
    scene_sim_result[:, 4] = np.mean(scene_result[:, kitchen2], 1)
    scene_sim_result[:, 5] = np.mean(scene_result[:, laun], 1)
    scene_sim_result[:, 6] = np.mean(scene_result[:, lr1], 1)
    scene_sim_result[:, 7] = np.mean(scene_result[:, lr2], 1)
    scene_sim_result[:, 8] = np.mean(scene_result[:, market], 1)
    scene_sim_result[:, 9] = np.mean(scene_result[:, pub], 1)


    np.save(result_scene_distance_dir, scene_sim_result)

    topic_id_num = 9219
    query_id = 'shot1_1'
    run_id = 'scene_0523_bow'
    for i in range(10):
        topic_id = str(topic_id_num+i)
        print(topic_id)
        if not os.path.exists(os.path.join(result_file_path, run_id, topic_id)):
            os.makedirs(os.path.join(result_file_path, run_id, topic_id))

        output = open(os.path.join(result_file_path, run_id, topic_id, 'TRECVID2013_11.res'), 'w')

        if i == 1:
            continue
        scene_sim = scene_sim_result[:, i]
        scene_single_result = np.argsort(-scene_sim)
        for j in range(1000):
            shot_id = 'shot' + str(shot_index[scene_single_result[j], 0]) + '_' + str(shot_index[scene_single_result[j], 1])
            shot_score = str(scene_sim[scene_single_result[j]])
            output.write(shot_id + ' #$# ' + query_id + ' #$# ' + shot_score + '\n')

        output.close()


if __name__ == '__main__':
    main()

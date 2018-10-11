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

    run_id = '0614_facenet_dir_fusion_mutual'
    query_id = 'shot1_1'

    ###############################################################################
    # topic info []
    chelsea = 0                      # 14000
    darrin = 1                       # 5400
    garry = 2                        # 4600
    heather = 3                      # 10800
    jack = 4                         # 13600
    jane = 5                         # 18000
    max = 6                          # 18000
    minty = 7                        # 6000
    mo = 8                           # 6000
    zainab = 9                       # 14000

    cafe1 = 0                        # 4000
    cafe2 = 1                        # 4000
    foyer = 2                        # 4000
    kitchen1 = 3                     # 4000
    kitchen2 = 4                     # 4000
    laun = 5                         # 4000
    LR1 = 6                          # 3000
    LR2 = 7                          # 2000 +
    market = 8                       # 4000
    pub = 9                          # 4000
    topics = np.zeros((30, 2), dtype=int)
    topics[:,:] = [[jane, cafe2],
                   [jane, pub],
                   [jane, market],
                   [chelsea, cafe2],
                   [chelsea, pub],
                   [chelsea, market],
                   [minty, cafe2],
                   [minty, pub],
                   [minty, market],
                   [garry, cafe2],
                   [garry, pub],
                   [garry, laun],
                   [mo, cafe2],
                   [mo, pub],
                   [mo, laun],
                   [darrin, cafe2],
                   [darrin, pub],
                   [darrin, laun],
                   [zainab, cafe2],
                   [zainab, laun],
                   [zainab, market],
                   [heather, cafe2],
                   [heather, laun],
                   [heather, market],
                   [jack, pub],
                   [jack, laun],
                   [jack, market],
                   [max, cafe2],
                   [max, laun],
                   [max, market]]
    topic_start_id = 9219
    result_file_path = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/result/tv2018/test2018'
    shot_meta_file = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/meta/Index.mat'
    mat_data = scio.loadmat(shot_meta_file)
    shot_index = mat_data['Index']

    gallery_num = 471526
    scene_num = 10
    person_num = 10
    ###############################################################################



    ###############################################################################
    # seperate result load and pre-process
    person_result_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/face_distance_0522.npy'
    scene_result_file  = '/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir-dba/distance_full.npy'
    scene_result_index_file = '/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/search-results/dir-dba/index_full.npy'

    # person
    person_distance  = np.load(person_result_file)
    person_distance[person_distance == 0] = 2
    person_similarity = np.exp(-person_distance)
    person_similarity = preprocessing.normalize(person_similarity, norm='l2')
    print('person score loaded')

    # dir scene
    scene_distance_original = np.load(scene_result_file)
    scene_distance_original_index = np.load(scene_result_index_file)
    for i in range(90):
        sort_index = np.argsort(scene_distance_original_index[i, :])
        scene_distance_original[i, :] = scene_distance_original[i, sort_index]
    scene_distance = np.zeros((gallery_num, scene_num), dtype=float)
    scene_distance[:, 0] = np.min(scene_distance_original[0:12, :], 0)   # 12  cafe1
    scene_distance[:, 1] = np.min(scene_distance_original[12:24, :], 0)  # 12  cafe2
    scene_distance[:, 2] = np.min(scene_distance_original[24:30, :], 0)  # 6   foyer
    scene_distance[:, 3] = np.min(scene_distance_original[30:36, :], 0)  # 6   kitchen1
    scene_distance[:, 4] = np.min(scene_distance_original[36:42, :], 0)  # 6   kitchen2
    scene_distance[:, 5] = np.min(scene_distance_original[42:54, :], 0)  # 12  laun
    scene_distance[:, 6] = np.min(scene_distance_original[54:60, :], 0)  # 6   LR1
    scene_distance[:, 7] = np.min(scene_distance_original[60:66, :], 0)  # 6   LR2
    scene_distance[:, 8] = np.min(scene_distance_original[66:78, :], 0)  # 12  market
    scene_distance[:, 9] = np.min(scene_distance_original[78:90, :], 0)  # 12  pub
    scene_similarity = np.exp(-scene_distance)
    scene_similarity = preprocessing.normalize(scene_similarity, norm='l2')

    # # bow scene
    # temp_scene = scene_similarity[:, 1]
    # temp_scene = temp_scene[np.argsort(-temp_scene)]
    # temp = temp_scene[30000]
    # scene_similarity[scene_similarity[:, 1] < temp, 1] = 0
    # scene_result_file_bow = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/scene_similarity_0523.npy'
    # scene_similarity_bow = np.load(scene_result_file_bow)
    # scene_similarity_bow = preprocessing.normalize(scene_similarity_bow, norm='l2')
    # scene_similarity_bow[:, 1] = 1.0
    # scene_similarity = scene_similarity * scene_similarity_bow


    # score adjust
    time_knn  = 3
    scene_threshold = 3000
    scene_similarity_knn = np.zeros((gallery_num, scene_num), dtype=float)
    for i in range(scene_num):
        temp_scene = scene_similarity[:, i]                     #
        temp_scene = temp_scene[np.argsort(-temp_scene)]        #
        temp = temp_scene[scene_threshold]                      #
        temp_scene = scene_similarity[:, i]
        temp_scene[temp_scene < temp] = 0                       #
        for j in range(gallery_num):
            if temp_scene[j] > 0:
                if j+1+time_knn < gallery_num:
                    temp_temp = np.max(temp_scene[j+1:j+1+time_knn])
                    if temp_temp > 0:
                        temp_scene[j + 1] = temp_temp
                elif (j+1+time_knn >= gallery_num and j + 1 < gallery_num):
                    temp_temp = np.max(temp_scene[j+1:gallery_num])
                    if temp_temp > 0:
                        temp_scene[j + 1] = temp_temp
        scene_similarity_knn[:, i] = temp_scene.T

    time_knn  = 2
    # person_threshold = [14000, 5400, 4600, 10800, 13600, 18000, 18000, 6000, 6000, 14000]
    person_threshold = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]
    person_similarity_knn = np.zeros((gallery_num, person_num), dtype=float)
    for i in range(person_num):
        temp_person = person_similarity[:, i]
        temp_person = temp_person[np.argsort(-temp_person)]
        temp = temp_person[person_threshold[i]]
        temp_person = person_similarity[:, i]
        temp_person[temp_person < temp] = 0
        for j in range(gallery_num):
            if temp_person[j] > 0:
                if j+1+time_knn < gallery_num:
                    temp_temp = np.max(temp_person[j+1:j+1+time_knn])
                    if temp_temp > 0:
                        temp_person[j + 1] = temp_person[j]
                elif (j+1+time_knn >= gallery_num and j + 1 < gallery_num):
                    temp_temp = np.max(temp_person[j+1:gallery_num])
                    if temp_temp > 0:
                        temp_person[j + 1] = temp_person[j]
        person_similarity_knn[:, i] = temp_person.T
    ###############################################################################


    # certain scene
    certain_scene_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/del/certain_scene.npy'
    if os.path.isfile(certain_scene_file):
        certain_scene = np.load(certain_scene_file)
    else:
        certain_scene = np.zeros((gallery_num, scene_num), dtype=int)
        for i in range(10):
            temp_scene = scene_similarity[:, i]
            temp_scene = temp_scene[np.argsort(-temp_scene)]
            temp = temp_scene[3000]
            temp_scene = scene_similarity[:, i]
            temp_scene[temp_scene < temp] = 0
            temp_scene[temp_scene >= temp] = 1
            certain_scene[:, i] = temp_scene.T
        np.save(certain_scene_file, certain_scene)
    print('certain scene loaded')

    # shot0_ shots
    shot0_shots_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/del/shot0_shots.npy'
    if os.path.isfile(shot0_shots_file):
        shot0_shots = np.load(shot0_shots_file)
    else:
        shot0_shots = np.zeros(gallery_num, dtype=int)
        shot0_shots[np.where(shot_index[:, 0] == 0)[0]] = 1
        np.save(shot0_shots_file, shot0_shots)
    print('shot0_ shots loaded')

    # outdoor scene, should be used carefully for pub
    outdoor_scene_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/del/outdoor_scene.npy'
    if os.path.isfile(outdoor_scene_file):
        outdoor_scene = np.load(outdoor_scene_file)
    else:
        outdoor_scene = np.zeros(gallery_num, dtype=int)
        outdoor_mat_data_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/del/outdoor_2016.mat'
        mat_data = scio.loadmat(outdoor_mat_data_file)
        outdoor_list = mat_data['outdoor']
        cell_num = len(outdoor_list)
        for i in range(cell_num):
            shot_id = outdoor_list[i][0][0]
            index_a = shot_id.find('shot')
            index_b = shot_id.find('_')
            shot_id_num_1 = shot_id[index_a + 4:index_b]
            shot_id_num_2 = shot_id[index_b + 1:]
            shot_position = list(set(np.where(shot_index[:, 0] == int(shot_id_num_1))[0]).intersection(
                set(np.where(shot_index[:, 1] == int(shot_id_num_2))[0])))[0]
            outdoor_scene[shot_position] = 1
        np.save(outdoor_scene_file, outdoor_scene)
    print('outdoor scene loaded')
    ###############################################################################



    ###############################################################################
    # fusion
    for i in range(topics.shape[0]):
        topic_id = topic_start_id + i
        print(topic_id, topics[i,0], topics[i,1])
        temp_person = person_similarity[:, topics[i, 0]]
        temp_scene  = scene_similarity[:, topics[i, 1]]
        temp_person_knn = person_similarity_knn[:, topics[i, 0]]
        temp_scene_knn  = scene_similarity_knn[:, topics[i, 1]]


        del_scene = np.delete(certain_scene, topics[i, 1], axis = 1)
        del_scene = np.sum(del_scene, axis=1)
        del_result = shot0_shots + del_scene
        if topics[i, 1] != pub:
            del_result = outdoor_scene + del_result
        del_result[del_result == 0] = -1
        del_result[del_result > 0] = 0
        del_result[del_result == -1] = 1.0

        final_similarity = temp_person + temp_scene #0.6*(temp_person + temp_scene) + 0.4*(temp_person_knn + temp_scene_knn)
        final_similarity_knn = temp_person_knn * temp_scene_knn # temp_person + temp_scene +
        # final_similarity_knn = final_similarity_knn * del_result

        final_similarity_knn[final_similarity_knn > 0] = 1.0
        final_similarity = final_similarity * final_similarity_knn

        print(len(np.where(final_similarity>0)[0]))
        final_similarity_result = np.argsort(-final_similarity)

        if not os.path.exists(os.path.join(result_file_path, run_id, str(topic_id))):
            os.makedirs(os.path.join(result_file_path, run_id, str(topic_id)))
        output = open(os.path.join(result_file_path, run_id, str(topic_id), 'TRECVID2013_11.res'), 'w')

        shot_list = np.empty((1000, 1), dtype=object)
        for j in range(1000):
            shot_id = 'shot' + str(shot_index[final_similarity_result[j], 0]) + '_' + str(shot_index[final_similarity_result[j], 1])
            shot_score = str(final_similarity[final_similarity_result[j]])
            output.write(shot_id + ' #$# ' + query_id + ' #$# ' + shot_score + '\n')
            shot_list[j, 0] = str(topic_id)+shot_id
        output.close()

        if not os.path.exists(os.path.join(result_file_path, run_id)):
            os.makedirs(os.path.join(result_file_path, run_id))
        scio.savemat(os.path.join(result_file_path, run_id, str(topic_id) + '.mat'), {'result': shot_list})


if __name__ == '__main__':
    main()





    # for i in range(10):
    #     topic_id = topic_start_id + i
    #     temp_scene  = scene_similarity[:, i]
    #     final_similarity_knn = temp_scene
    #     final_similarity_result = np.argsort(-final_similarity_knn)
    #
    #     if not os.path.exists(os.path.join(result_file_path, run_id, str(topic_id))):
    #         os.makedirs(os.path.join(result_file_path, run_id, str(topic_id)))
    #     output = open(os.path.join(result_file_path, run_id, str(topic_id), 'TRECVID2013_11.res'), 'w')
    #
    #     for j in range(3000, 4000):
    #         shot_id = 'shot' + str(shot_index[final_similarity_result[j], 0]) + '_' + str(shot_index[final_similarity_result[j], 1])
    #         shot_score = str(final_similarity_knn[final_similarity_result[j]])
    #         output.write(shot_id + ' #$# ' + query_id + ' #$# ' + shot_score + '\n')
    #     output.close()
    # for i in range(10):
    #     print(len(np.where(scene_similarity[:, i]>0)[0]))
    # print('scene score loaded')
    #
    #
    # return
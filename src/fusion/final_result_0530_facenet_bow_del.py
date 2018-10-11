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


def main():

    person_result_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/face_distance_0522.npy'
    scene_result_file  = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/scene_similarity_0523.npy'
    result_file_path = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/result/tv2018/test2018'
    shot_meta_file = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/meta/Index.mat'
    mat_data = scio.loadmat(shot_meta_file)
    shot_index = mat_data['Index']

    gallery_num = 471526
    del_result = np.zeros(gallery_num, dtype=int)
    del_result = del_result + 1
    del_result[np.where(shot_index[:, 0] == 0)[0]] = 0
    print('delete all shot0_ shots')

    del_noface_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/del/noface_2016.mat'
    del_outdoor_file = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/del/outdoor_2017.mat'
    mat_data = scio.loadmat(del_noface_file)
    del_noface_list = mat_data['noface']
    mat_data = scio.loadmat(del_outdoor_file)
    del_outdoor_list = mat_data['outdoor']

    # cell_num = len(del_noface_list)
    # for i in range(cell_num):
    #     shot_id = del_noface_list[i][0][0]
    #     index_a = shot_id.find('shot')
    #     index_b = shot_id.find('_')
    #     shot_id_num_1 = shot_id[index_a + 4:index_b]
    #     shot_id_num_2 = shot_id[index_b + 1:]
    #     shot_position = list(set(np.where(shot_index[:, 0] == int(shot_id_num_1))[0]).intersection(
    #         set(np.where(shot_index[:, 1] == int(shot_id_num_2))[0])))[0]
    #     del_result[shot_position] = 0
    # print('delete all no face shots')

    cell_num = len(del_outdoor_list)
    for i in range(cell_num):
        shot_id = del_outdoor_list[i][0][0]
        index_a = shot_id.find('shot')
        index_b = shot_id.find('_')
        shot_id_num_1 = shot_id[index_a + 4:index_b]
        shot_id_num_2 = shot_id[index_b + 1:]
        shot_position = list(set(np.where(shot_index[:, 0] == int(shot_id_num_1))[0]).intersection(
           set(np.where(shot_index[:, 1] == int(shot_id_num_2))[0])))[0]
        del_result[shot_position] = 0
    print('delete all outdoor shots')


    chelsea = 0
    darrin = 1
    garry = 2
    heather = 3
    jack = 4
    jane = 5
    max = 6
    minty = 7
    mo = 8
    zainab = 9

    cafe1 = 0
    cafe2 = 1
    foyer = 2
    kitchen1 = 3
    kitchen2 = 4
    laun = 5
    LR1 = 6
    LR2 = 7
    market = 8
    pub = 9

    topics = np.zeros((30, 2), dtype=int)
    topics[:,:] = [[jane, cafe2], [jane, pub], [jane, market],
              [chelsea, cafe2], [chelsea, pub], [chelsea, market],
              [minty, cafe2], [minty, pub], [minty, market],
              [garry, cafe2], [garry, pub], [garry, laun],
              [mo, cafe2], [mo, pub], [mo, laun],
              [darrin, cafe2], [darrin, pub], [darrin, laun],
              [zainab, cafe2], [zainab, laun], [zainab, market],
              [heather, cafe2], [heather, laun], [heather, market],
              [jack, pub], [jack, laun], [jack, market],
              [max, cafe2], [max, laun], [max, market]]
    topic_start_id = 9219


    person_distance  = np.load(person_result_file)
    person_distance[person_distance == 0] = 2
    person_similarity = 1 / person_distance
    person_similarity = preprocessing.normalize(person_similarity, norm='l2')
    scene_similarity = np.load(scene_result_file)
    scene_similarity = preprocessing.normalize(scene_similarity, norm='l2')

    run_id = '0530_facenet_bow_del'
    query_id = 'shot1_1'
    for i in range(topics.shape[0]):
        topic_id = topic_start_id + i
        print(topic_id, topics[i,0], topics[i,1])

        # final_similarity = person_similarity[:, topics[i,0]] + scene_similarity[:, topics[i,1]]
        final_similarity = np.multiply(person_similarity[:, topics[i,0]], scene_similarity[:, topics[i,1]])
        final_similarity = np.multiply(final_similarity, del_result)
        final_similarity_result = np.argsort(-final_similarity)

        if not os.path.exists(os.path.join(result_file_path, run_id, str(topic_id))):
            os.makedirs(os.path.join(result_file_path, run_id, str(topic_id)))
        output = open(os.path.join(result_file_path, run_id, str(topic_id), 'TRECVID2013_11.res'), 'w')

        for j in range(1000):
            shot_id = 'shot' + str(shot_index[final_similarity_result[j], 0]) + '_' + str(shot_index[final_similarity_result[j], 1])
            shot_score = str(final_similarity[final_similarity_result[j]])
            output.write(shot_id + ' #$# ' + query_id + ' #$# ' + shot_score + '\n')

        output.close()




if __name__ == '__main__':
    main()

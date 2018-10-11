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

    run_id = '0531_facenet_bow_extend'
    query_id = 'shot1_1'

    gallery_num = 471526
    person_similarity_knn = np.zeros((gallery_num, 15), dtype=np.float)
    scene_similarity_knn = np.zeros((gallery_num, 15), dtype=np.float)

    for i in range(topics.shape[0]):
        topic_id = topic_start_id + i
        print(topic_id, topics[i,0], topics[i,1])

        final_similarity = person_similarity[:, topics[i,0]] + scene_similarity[:, topics[i,1]]

        person_similarity_knn[:, 0] = person_similarity[:, topics[i, 0]]
        scene_similarity_knn[:, 0] = scene_similarity[:, topics[i,1]]
        for j in range(7):
            person_similarity_knn[0:gallery_num-j-1, j+1] = person_similarity[j+1:gallery_num, topics[i, 0]]
            scene_similarity_knn[0:gallery_num-j-1, j+1] = scene_similarity[j+1:gallery_num, topics[i, 1]]
        for j in range(7):
            person_similarity_knn[j+1:gallery_num, j+7] = person_similarity[0:gallery_num-j-1, topics[i, 0]]
            scene_similarity_knn[j+1:gallery_num, j+7] = scene_similarity[0:gallery_num-j-1, topics[i, 1]]

        person_similarity_knn_final = person_similarity_knn.max(axis=1)
        scene_similarity_knn_final = scene_similarity_knn.max(axis=1)
        person_similarity_knn_final[person_similarity_knn_final < 0.5 ] = 0
        scene_similarity_knn_final[scene_similarity_knn_final < 0.5] = 0
        final_similarity_knn = person_similarity_knn_final + scene_similarity_knn_final

        final_similarity_result = np.argsort(-final_similarity_knn)

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

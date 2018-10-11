from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
sys.path.append("./facenet-master/src/")

import os
import argparse
import numpy as np
import random
from time import sleep
from sklearn.metrics.pairwise import euclidean_distances


def main():

    result_face_distance_dir = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/face_distance_0522.npy'
    gallery_face_feature_dir = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/data/face_gallery/'
    shot_list_file = '/net/per610a/export/das11g/caizhizhu/ins/ins2013/frames_png/clips.txt'
    dict_face_feature_dir = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/data/face_dict/dict_feature.npy'

    gallery_num = 471526
    person_num  = 10

    chelsea_start_index = 0
    chelsea_end_index = 10
    darrin_start_index = 10
    darrin_end_index = 20
    garry_start_index = 20
    garry_end_index = 30
    heather_start_index = 30
    heather_end_index = 40
    jack_start_index = 40
    jack_end_index = 50
    jane_start_index = 50
    jane_end_index = 60
    max_start_index = 60
    max_end_index = 70
    minty_start_index = 70
    minty_end_index = 80
    mo_start_index = 80
    mo_end_index = 90
    zainab_start_index = 90
    zainab_end_index = 100

    face_result = np.zeros((gallery_num, person_num), dtype=float)

    i = 0
    dict_feature = np.load(dict_face_feature_dir)
    # print(dict_feature)
    print(dict_feature.shape)
    dict_feature = dict_feature[0:100, :]
    shot_id_list_file = open(shot_list_file, 'r')
    for shot_id in shot_id_list_file.readlines():
        shot_id = shot_id.strip('\n')
        shot_id = shot_id.strip()
        gallery_feature_file = os.path.join(gallery_face_feature_dir, shot_id+'.npy')
        if os.path.exists(gallery_feature_file):
            gallery_feature = np.load(gallery_feature_file)
            if gallery_feature != []:
                distance_matrix = euclidean_distances(gallery_feature, dict_feature)
                face_result[i, 0] = distance_matrix[:, chelsea_start_index:chelsea_end_index].min()
                face_result[i, 1] = distance_matrix[:, darrin_start_index:darrin_end_index].min()
                face_result[i, 2] = distance_matrix[:, garry_start_index:garry_end_index].min()
                face_result[i, 3] = distance_matrix[:, heather_start_index:heather_end_index].min()
                face_result[i, 4] = distance_matrix[:, jack_start_index:jack_end_index].min()
                face_result[i, 5] = distance_matrix[:, jane_start_index:jane_end_index].min()
                face_result[i, 6] = distance_matrix[:, max_start_index:max_end_index].min()
                face_result[i, 7] = distance_matrix[:, minty_start_index:minty_end_index].min()
                face_result[i, 8] = distance_matrix[:, mo_start_index:mo_end_index].min()
                face_result[i, 9] = distance_matrix[:, zainab_start_index:zainab_end_index].min()
                # print(distance_matrix.shape)
            else:
                print('null')
        else:
            print(shot_id)

        i = i + 1
        print(i)

    np.save(result_face_distance_dir, face_result)
    print("Done")


if __name__ == '__main__':
    main()

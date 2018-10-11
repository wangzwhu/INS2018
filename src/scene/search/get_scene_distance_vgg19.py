from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import scipy.io as scio
import sys
import os
import argparse
import numpy as np
import random
from time import sleep
from sklearn.metrics.pairwise import euclidean_distances
import h5py
from multiprocessing import Pool


def calculate_distance(line_id):
    shot_id = alllines[line_id]
    shot_id = shot_id.strip('\n')
    shot_id = shot_id.strip()
    gallery_feature_file = os.path.join(gallery_scene_feature_dir, shot_id + '.mat')
    if os.path.exists(gallery_feature_file):
        try:
            f = h5py.File(gallery_feature_file, 'r')
        except:
            gallery_feature = scio.loadmat(gallery_feature_file)['clip_cnn_feat'].T
        else:
            gallery_feature = f['clip_cnn_feat'][:]
            f.close()

        if gallery_feature != []:
            scene_result[line_id, 0] = np.min(euclidean_distances(gallery_feature, query_data_0))
            scene_result[line_id, 1] = np.min(euclidean_distances(gallery_feature, query_data_1))
            scene_result[line_id, 2] = np.min(euclidean_distances(gallery_feature, query_data_2))
            scene_result[line_id, 3] = np.min(euclidean_distances(gallery_feature, query_data_3))
            scene_result[line_id, 4] = np.min(euclidean_distances(gallery_feature, query_data_4))
            scene_result[line_id, 5] = np.min(euclidean_distances(gallery_feature, query_data_5))
            scene_result[line_id, 6] = np.min(euclidean_distances(gallery_feature, query_data_6))
            scene_result[line_id, 7] = np.min(euclidean_distances(gallery_feature, query_data_7))
            scene_result[line_id, 8] = np.min(euclidean_distances(gallery_feature, query_data_8))
            scene_result[line_id, 9] = np.min(euclidean_distances(gallery_feature, query_data_9))
        else:
            print('null')
    else:
        print(shot_id)
    print(line_id)


if __name__ == '__main__':

    result_scene_distance_dir = '/net/dl380g7a/export/ddn11c1/wangz/ins2018/distance/scene_distance_vgg19_0602.npy'
    gallery_scene_feature_dir = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/feature/keyframe-5/tv2017/test2017/cnn_vgg19_mat/'
    query_scene_feature_dir = '/net/dl380g7a/export/ddn11a2/ledduy/kaori-visualsearch/kaori-ins16/feature/keyframe-0/tv2017/query2017/cnn_vgg19_mat/location/'
    shot_list_file = '/net/per610a/export/das11g/caizhizhu/ins/ins2013/frames_png/clips.txt'

    gallery_num = 471526
    scene_num   = 10
    scene_result = np.zeros((gallery_num, scene_num), dtype=float)

    scene_list = ['cafe1', 'cafe2', 'foyer', 'kitchen1', 'kitchen2', 'laun', 'LR1', 'LR2', 'market', 'pub']

    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[0] + '.mat'), 'r')
    query_data_0 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[1] + '.mat'), 'r')
    query_data_1 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[2] + '.mat'), 'r')
    query_data_2 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[3] + '.mat'), 'r')
    query_data_3 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[4] + '.mat'), 'r')
    query_data_4 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[5] + '.mat'), 'r')
    query_data_5 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[6] + '.mat'), 'r')
    query_data_6 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[7] + '.mat'), 'r')
    query_data_7 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[8] + '.mat'), 'r')
    query_data_8 = f['all_feat'][:]
    f.close()
    f = h5py.File(os.path.join(query_scene_feature_dir, scene_list[9] + '.mat'), 'r')
    query_data_9 = f['all_feat'][:]
    f.close()

    shot_id_list_file = open(shot_list_file, 'r')
    alllines = shot_id_list_file.readlines()

    pool = Pool()
    pool.map(calculate_distance, range(gallery_num))
    pool.close()

    np.save(result_scene_distance_dir, scene_result)
    print("Done")

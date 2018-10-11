from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
sys.path.append("./facenet-master/src/")

import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep

def main():
    start_line = 0 # 471526
    end_line   = 50000
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_dir  = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/src/facenet-master/model/'
    key_frames_path = '/net/per610a/export/das11g/caizhizhu/ins/ins2013/frames_png/'
    output_feature_dir = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/data/face_gallery/'
    shot_list_file  = 'clips.txt'
    frame_list_file = 'frames.txt'

    detect_multiple_faces = True
    margin = 44
    image_size = 182
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    print('Creating networks and loading parameters')
    g1 = tf.Graph() # detect and align
    g2 = tf.Graph() # feature
    with g1.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess1 = tf.Session(graph=g1)
        with sess1.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess1, None)
    with g2.as_default():
        sess2 = tf.Session(graph=g2)
        with sess2.as_default():
            facenet.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

    output_feature_dir = os.path.expanduser(output_feature_dir)
    if not os.path.exists(output_feature_dir):
        os.makedirs(output_feature_dir)

    shot_id_list_file = open(os.path.join(key_frames_path, shot_list_file), 'r')
    for shot_id in shot_id_list_file.readlines()[start_line:end_line]:
        shot_id = shot_id.strip('\n')
        shot_id = shot_id.strip()

        shot_face_feature = None
        shot_face_num = 0

        frame_id_list_file = open(os.path.join(key_frames_path, shot_id, frame_list_file), 'r')
        for frame_id in frame_id_list_file.readlines():
            frame_id = frame_id.strip('\n')
            frame_id = frame_id.strip()

            img = misc.imread(os.path.join(key_frames_path, shot_id, frame_id))
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:, :, 0:3]

            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]

                if nrof_faces > 1:
                    if detect_multiple_faces:
                        for i in range(nrof_faces):
                            det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        img_center = img_size / 2
                        offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                             (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        index = np.argmax(
                            bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                        det_arr.append(det[index, :])
                else:
                    det_arr.append(np.squeeze(det))

                # for each detected face
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    scaled = misc.imresize(scaled, (160, 160), interp='bilinear')

                    # output_filename = os.path.join(output_feature_dir, shot_id + '.' + frame_id + '.png')
                    # filename_base, file_extension = os.path.splitext(output_filename)
                    # if detect_multiple_faces:
                    #    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                    # else:
                    #    output_filename_n = "{}{}".format(filename_base, file_extension)
                    # misc.imsave(output_filename_n, scaled)

                    scaled_reshape = []
                    pre_img = facenet.prewhiten(scaled)
                    scaled_reshape.append(pre_img.reshape(-1, 160, 160, 3))
                    emb_temp = np.zeros((1, embedding_size))
                    emb_temp[0, :] = sess2.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False})[0]

                    if shot_face_num == 0:
                        shot_face_feature = emb_temp[0, :]
                    else:
                        shot_face_feature = np.concatenate((shot_face_feature, emb_temp[0, :]), axis=0)
                    shot_face_num = shot_face_num + 1
        frame_id_list_file.close()

        if shot_face_num > 0 :
            shot_face_feature = shot_face_feature.reshape(shot_face_num, embedding_size)

        print(shot_id)
        np.save(os.path.join(output_feature_dir, shot_id + '.npy'), shot_face_feature)

    shot_id_list_file.close()





if __name__ == '__main__':
    main()

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    model_dir  = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/src/facenet-master/model/'
    image_list_file_dir = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/face_dict/dict.txt'
    input_dir  = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/face_dict/'
    output_dir = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/aligned_face_dict/'
    output_feature_dir = '/net/per920a/export/das14a/satoh-lab/wangz/ins2018/data/face_dict/dict_feature.npy'

    detect_multiple_faces = False
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


    print('loading image path')
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_paths = facenet.get_image_paths(input_dir)

    # emb_feature = np.zeros((len(image_paths), embedding_size))
    emb_feature = np.zeros((100, embedding_size))

    img_id = 0
    image_list_file = open(image_list_file_dir, 'r')
    for image_path in image_list_file.readlines():
        image_path = image_path.strip('\n')
        image_path = image_path.strip()

        img = misc.imread(input_dir + image_path)
        # throw small image, and change gray image to color image
        if img.ndim < 2:
            print('Unable to align "%s"' % image_path)
            continue
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]
        # face detection
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

                # filename = os.path.splitext(os.path.split(image_path)[1])[0]
                # output_filename = os.path.join(output_dir, filename + '.png')
                # filename_base, file_extension = os.path.splitext(output_filename)
                # if detect_multiple_faces:
                #    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                # else:
                #     output_filename_n = "{}{}".format(filename_base, file_extension)
                # misc.imsave(output_filename_n, scaled)

                scaled_reshape = []
                pre_img = facenet.prewhiten(scaled)
                scaled_reshape.append(pre_img.reshape(-1, 160, 160, 3))
                emb_temp = np.zeros((1, embedding_size))
                emb_temp[0, :] = sess2.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False})[0]
                emb_feature[img_id, :] = emb_temp[0, :]
                img_id = img_id+1
        else:
            print('Unable to align "%s"' % image_path)

    np.save(output_feature_dir, emb_feature)

if __name__ == '__main__':
    main()

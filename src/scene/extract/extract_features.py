import numpy as np
import argparse
import os
import sys
import time
import joblib
import glob
import csv
from tqdm import tqdm
from sklearn.preprocessing import normalize

sys.path.insert(0, "/home/hinami/work/obd/py-faster-rcnn/caffe-fast-rcnn/python")
sys.path.insert(0, "/home/hinami/work/deep_retrieval/")
import caffe
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_image_dir', type=str, required=True,
            help="Directory to the query images.")
    parser.add_argument('--gallery_image_dir', type=str, required=True,
            help="Directory to the gallery key frames.")
    parser.add_argument('--query_feature_dir', type=str, required=True,
            help="Directory where query features will be written to.")
    parser.add_argument('--gallery_feature_dir', type=str, required=True,
            help="Directory where gallery features will be written to.")
    parser.add_argument('--gpu', type=int, required=True,
            help='GPU ID to use (e.g. 0)')
    parser.add_argument('--S', type=int, default=800,
            help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--L', type=int, default=2,
            help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--step', type=int, default=50000)
    parser.add_argument('--proto', type=str,
            default="/home/hinami/work/deep_retrieval/deploy_resnet101_normpython.prototxt",
            help='Path to the prototxt file')
    parser.add_argument('--weights', type=str,
            default="/home/hinami/work/deep_retrieval/model.caffemodel",
            help='Path to the caffemodel file')
    parser.add_argument('--aggregate', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    print("Args: {}".format(args))
    return args

class ImageHelper:
    def __init__(self, S, L, means):
        self.S = S
        self.L = L
        self.means = means

    def prepare_image_and_grid_regions_for_network(self, fname, roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        I, im_resized = self.load_and_prepare_image(fname, roi)
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = []
            all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
            R = self.pack_regions_for_network(all_regions)
        return I, R

    def get_rmac_features(self, I, R, net, aggregate=True):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        key = "rmac/normalized" if aggregate else "pooled_rois/pca/normalized"
        net.forward(end=key)
        return np.squeeze(net.blobs[key].data)

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S)/np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means
        return I, im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R

    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)


def load_dir_model(proto_path, weights_path, gpu):
    # Configure caffe and load the network
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    print(proto_path, weights_path)
    net = caffe.Net(proto_path, weights_path, caffe.TEST)
    # from IPython import embed; embed()
    return net

def extract_dir_feature(model, impath, image_helper, scales=[550, 800, 1050], aggregate=True):
    """ Extract DIR feature from one image with multiple scale
    model: DIR model (caffe.Net)
    impath: path to an image
    image_helper: image helper
    scales: image scales  [550, 800, 1050] is used in the official code
    aggregate: aggregate region features or not
        If True, global features are output
        If False, all region features are output
    """
    features = []
    for S in scales:
        # Set the scale of the image helper
        image_helper.S = S
        I, R = image_helper.prepare_image_and_grid_regions_for_network(
                impath, roi=None)
        features.append(image_helper.get_rmac_features(I, R, model, aggregate))
    if aggregate:
        feature = np.dstack(features).sum(axis=2)
        feature = normalize(feature.reshape(1,-1), axis=1, norm='l2')[0]
        return feature.astype(np.float32)
    else:
        return np.concatenate(features).astype(np.float32)

def get_query_image_paths(args):
    query_image_paths = [os.path.join(args.query_image_dir, image_name)
            for image_name in glob.glob(os.path.join(args.query_image_dir, '*.bmp'))]
    print('found {} query images.'.format(len(query_image_paths)))
    return query_image_paths

def get_gallery_image_folders_paths(args):
    gallery_image_folders_paths = [os.path.join(args.gallery_image_dir, folder_name)
            for folder_name in glob.glob(os.path.join(args.gallery_image_dir,'shot*'))]
    print('found {} gallery folders.'.format(len(gallery_image_folders_paths)))
    return gallery_image_folders_paths

def extract_query_features(args):
    image_paths = get_query_image_paths(args)
    num_images = len(image_paths)
    if not os.path.exists(args.query_feature_dir):
        print('{} not exsit, it will be created.'.format(args.query_feature_dir))
        os.makedirs(args.query_feature_dir)
    for impath in image_paths:
        featpath = os.path.join(args.query_feature_dir,
                os.path.splitext(os.path.basename(impath))[0] + ".jbl")
        if os.path.exists(featpath):
            print("{} already exists".format(featpath))
            continue
        feats = extract_dir_feature(model, impath, image_helper, scales, args.aggregate)
        joblib.dump({"feats":feats.astype(np.float32), "locs": None}, featpath, compress=3)
        if args.verbose:
            print("{} ({})".format(impath, feats.shape))

def extract_gallery_features(args):
    image_folders_paths = get_gallery_image_folders_paths(args)
    for single_floder_path in tqdm(image_folders_paths[args.start:args.start+args.step]):
        # if folder exsits, extract features
        image_paths = [os.path.join(single_floder_path, image_path)
                for image_path in glob.glob(os.path.join(single_floder_path, '*.png'))]
        featpath = os.path.join(args.gallery_feature_dir,
                single_floder_path.split('/')[-1] + ".jbl")
        if os.path.exists(featpath):
            print("{} already exists".format(featpath))
            continue
        features = []
        for impath in image_paths:
            single_feature = extract_dir_feature(model, impath, image_helper, scales, args.aggregate)
            features.append(single_feature)
        joblib.dump({"feats":np.array(features).astype(np.float32), "locs": None}, featpath, compress=3)
        if args.verbose:
            print("{} ({})".format(impath, feats.shape))

if __name__ == '__main__':
    args = parse_args()
    means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]
    image_helper = ImageHelper(800, 2, means)
    model = load_dir_model(args.proto, args.weights, args.gpu)
    scales = [800 - 250, 800, 800 + 250]
    print('start extracting query features.')
    extract_query_features(args)
    print('start extracting gallery features.')
    extract_gallery_features(args)

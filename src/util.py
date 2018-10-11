import os
import time
import glob
import numpy as np
import joblib
from joblib import Parallel, delayed


def load_features(feature_dir, feature_type, ext, cache_file=None, use_cache=True):
    if cache_file is None:
        cache_file = os.path.join(feature_dir, 'cache_features.npy')
    if use_cache and os.path.exists(cache_file):
        print("Loading {} ...".format(cache_file))
        return np.load(cache_file)
    else:
        if use_cache:
            print("Cache {} does not exists. ".format(cache_file))
        if ext == ".jbl":
            paths = get_feature_path_list(feature_dir, feature_type, ext)
            print('path', paths[-1])
            feats = Parallel(n_jobs=-1, verbose=5)(delayed(joblib.load)(path)
                    for path in paths)
            feats = np.vstack([feat["feats"] for feat in feats]).astype(np.float32)
            if use_cache:
                np.save(cache_file, feats)
        else:
            raise(Exception("Unknown extension: {}".format(ext)))
    print('Features loaded (shape={})'.format(feats.shape))
    return feats

def get_feature_path_list(feature_dir, feature_type, ext):
    '''
    Retrun a list of feature paths.

    Args:
        feature_dir: String. Directory of features.
        feature_type: String. Type of features. Choice=['query', 'database']
        ext: String. Extension of extracted image features. It starts with a dot, like '.delf', '.npy'.
    '''
    if feature_type == 'query':
        feature_names = glob.glob(os.path.join(feature_dir, '*'+ext))
        feature_path_list = [os.path.join(
            feature_dir, name) for name in feature_names]
    elif feature_type =='database':
        feature_names_list_file = '/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/clips.txt'
        with open(feature_names_list_file, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        feature_path_list = [os.path.join(
            feature_dir, name+ext) for name in feature_names]
    else:
        raise Exception('Non-defined feature type.')
    return feature_path_list

def new_expansion_featuers(query_feature_dir, new_query_dir, cache_file=None, use_cache=True):
    scene_names = get_scene_names()
    # load features
    if cache_file is None:
        cache_file = os.path.join(new_query_dir, 'cache_features.npy')
    if use_cache and os.path.exists(cache_file):
        query_features = np.load(cache_file)
        print('query features:', query_features)
        return query_features
    else:
        print('cache file does not exsit.')
        query_features = []
        for scene in scene_names:
            scene_feature_path = [os.path.join(query_feature_dir, name) for
                    name in glob.glob(os.path.join(query_feature_dir, scene+'*.jbl'))]
            scene_feature = np.array([joblib.load(feature_path)['feats'] for feature_path in scene_feature_path])
            mean_scene_feature = scene_feature.mean(axis=0)
            featpath = os.path.join(new_query_dir, scene+'.jbl')
            joblib.dump({"feats":np.array(mean_scene_feature).astype(np.float32), "locs": None}, featpath, compress=3)
            query_features.append(mean_scene_feature)
        query_features = np.vstack(query_features)
        np.save(cache_file, query_features)
        print('query features: {}'.format(query_features.shape))
    return query_features

def load_single_feature(feature_dir, feature_name):
    return joblib.load(os.path.join(feature_dir, feature_name+'.jbl'))['feats']

class KNN(object):
    """KNN class
    Args:
        dataset: vectors in dataset to be queired
        method: method to measure distance between vectors
        gpu_ids: indices of GPUs to use
    """
    def __init__(self, dataset, method, gpu_ids='0'):
        import faiss
        index = {'cosine': faiss.IndexFlatIP,
                 'euclidean': faiss.IndexFlatL2}[method](dataset.shape[1])
        if gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(dataset.astype(np.float32))
        self.index = index

    def search(self, queries, k):
        """Search knn
        Args:
            queries: query vectors
            k: get top-k results
        Returns:
            distances: cosine distances(similarities) between vectors
            indices: knn indices
        """
        distances, indices = self.index.search(queries, k)
        return (distances, indices)

def get_face_dict():
    return {'chelsea':0, 'darrin':1, 'garry':2, 'heather':3, 'jack':4,
            'jane':5, 'max':6, 'minty':7, 'mo':8, 'zainab':9}

def get_scene_names():
    return ['cafe1', 'cafe2', 'foyer', 'kitchen1', 'kitchen2', 'laun', 'LR1', 'LR2', 'market', 'pub']

def get_query_order():
    person_query_order = ['jane', 'jane', 'jane', 'chelsea', 'chelsea', 'chelsea',
            'minty', 'minty', 'minty', 'garry', 'garry', 'garry',
            'mo', 'mo', 'mo', 'darrin', 'darrin', 'darrin',
            'zainab', 'zainab', 'zainab', 'heather', 'heather', 'heather',
            'jack', 'jack', 'jack', 'max', 'max', 'max']
    scene_query_order = ['cafe2', 'pub', 'market', 'cafe2', 'pub', 'market',
            'cafe2', 'pub', 'market', 'cafe2', 'pub', 'laun',
            'cafe2', 'pub', 'laun','cafe2', 'pub', 'laun',
            'cafe2', 'laun', 'market', 'cafe2', 'laun', 'market',
            'pub', 'laun', 'market', 'cafe2', 'laun', 'market']
    return person_query_order, scene_query_order

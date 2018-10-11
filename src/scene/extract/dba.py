import os
import sys
import joblib
import argparse
import numpy as np
sys.path.append(os.getcwd())
from src.scene.util import get_feature_path_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--feature_dir',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir/gallery',
            help='Directory to features after DBA.'
            )
    parser.add_argument(
            '--feature_dba_dir',
            type=str,
            default='/net/per920a/export/das14a/satoh-lab/suaimin/trecvid/share/features/dir-dba',
            help='Directory to features after DBA.'
            )
    args = parser.parse_args()
    # get feature path list
    feature_path_list = get_feature_path_list(args.feature_dir,
            'database', '.jbl')
    print(feature_path_list[-1])
    for feature_path in feature_path_list:
        feature = joblib.load(feature_path)['feats']
        new_feature = np.mean(feature, axis=0)
        print(feature.shape, new_feature.shape)
        feature_name = os.path.basename(feature_path)
        joblib.dump({"feats":new_feature.astype(np.float32), "locs": None},
                os.path.join(args.feature_dba_dir, feature_name), compress=3)

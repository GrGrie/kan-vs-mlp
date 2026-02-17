import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from pathlib import Path

class MetaShiftCatsDogsDataset(WILDSDataset):
    _dataset_name = 'MetaShiftCatsDogs'

    def __init__(self, root_dir='', 
            split_scheme='official', test_pct = 0.2, val_pct = 0.1, data_seed = None):
        self._data_dir = Path(self.initialize_data_dir(root_dir))


        all_data = []
        dirs = {
            'train/cat/cat(indoor)': [1, 1],
            'train/dog/dog(outdoor)': [0, 0],
            'test/cat/cat(outdoor)': [1, 0],
            'test/dog/dog(indoor)': [0, 1]
        }
        for dir in dirs:
            folder_path = self._data_dir/dir
            y = dirs[dir][0]
            g = dirs[dir][1]
            for img_path in folder_path.glob('*.jpg'):
                all_data.append({
                    'path': img_path,
                    'y': y,
                    'g': g
                })
        
        df = pd.DataFrame(all_data)

        # Get the y values
        self._y_array = torch.LongTensor(df['y'].values)
        self._y_size = 1
        self._n_classes = 2

        # Get metadata
        self._metadata_array = torch.stack(
            (torch.LongTensor(df['g'].values), self._y_array),
            dim=1
        )
        self._metadata_fields = ['env', 'y']
        self._metadata_map = {
            'env': ['outdoor', ' indoor'], 
            'y': ['dog', 'cat']
        }

        self._original_resolution = (224, 224) # as images are different sizes, we resize everything to 224 x 224
        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['env', 'y']))
        
        self._split_scheme = split_scheme
        test_idxs = np.random.choice(np.arange(len(df)), size = int(len(df) * test_pct), replace = False)
        val_idxs = np.random.choice(np.setdiff1d(np.arange(len(df)), test_idxs), size = int(len(df) * val_pct), replace = False)
        self._split_array = np.zeros((len(df), 1))
        self._split_array[val_idxs] = 1
        self._split_array[test_idxs] = 2

        self.df = df

        super().__init__(self._data_dir, split_scheme)

    def get_input(self, idx):
       # Note: idx and filenames are off by one.
       img_filename =self.df.iloc[idx]['path']
       x = Image.open(img_filename).convert('RGB').resize((self._original_resolution))
       return x

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels 
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

# Modified from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/wilds_dataset.py

import os

import torch
import numpy as np
import pandas as pd

class WILDSDataset:
    """
    Shared dataset class for all WILDS datasets.
    Each data point in the dataset is an (x, y, metadata) tuple, where:
    - x is the input features
    - y is the target
    - metadata is a vector of relevant information, e.g., domain.
      For convenience, metadata also contains y.
    """
    DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
    DEFAULT_SPLIT_NAMES = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

    def __init__(self, root_dir, split_scheme):
        if len(self._metadata_array.shape) == 1:
            self._metadata_array = self._metadata_array.unsqueeze(1)
        self.check_init()

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        raise NotImplementedError

    def eval(self, y_pred, y_true, metadata):
        """
        Args:
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        raise NotImplementedError

    def get_subset(self, split, frac=1.0, transform=None, ordering=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        og_group_counts = None
        if '_train' in split:
            split_mask = self.split_array == self.split_dict['train']
            groups, group_counts = self._eval_grouper.metadata_to_group(
                self.metadata_array[split_mask.ravel()],
                return_counts=True)
            og_group_counts = group_counts
            indices = []
            n_groups = torch.count_nonzero(group_counts, dim=0)
            if split == 'ds_train': # downsample
                num_to_retain = [min([int(sum(groups == g)) for g in range(n_groups)])] * n_groups
            elif split == 'us_train':
                num_to_retain = [max([int(sum(groups == g)) for g in range(n_groups)])] * n_groups
            elif split == 'balanced_train': # balance to similar to validation set
                split_mask_val = self.split_array == self.split_dict['val']
                _, group_counts_val = self._eval_grouper.metadata_to_group(
                    self.metadata_array[split_mask_val.ravel()],
                    return_counts=True)
                val_portions = np.array(group_counts_val) / min(group_counts_val)
                min_g = min(group_counts)
                num_to_retain = [int(e) for e in (val_portions * min_g).tolist()]
            for g in range(n_groups):
                split_idx = np.where(split_mask)[0]
                split_idx = split_idx[np.where(groups == g)[0]]
                if num_to_retain[g] <= len(split_idx) :
                    indices.extend(split_idx.tolist()[:num_to_retain[g]])
                else:
                    split_idx = split_idx.tolist()
                    indices.extend(np.random.choice(split_idx, num_to_retain[g]).tolist())
            split_idx = np.sort(indices)
        elif split == 'ds_test':
            split_mask = self.split_array == self.split_dict['test']
            groups, group_counts = self._eval_grouper.metadata_to_group(
                self.metadata_array[split_mask.ravel()],
                return_counts=True)
            og_group_counts = group_counts
            indices = []
            n_groups = len(group_counts)
             # downsample
            num_to_retain = [min([int(sum(groups == g)) for g in range(n_groups)])] * n_groups
            
            for g in range(n_groups):
                split_idx = np.where(split_mask)[0]
                split_idx = split_idx[np.where(groups == g)[0]]
                if num_to_retain[g] <= len(split_idx) :
                    indices.extend(split_idx.tolist()[:num_to_retain[g]])
                else:
                    split_idx = split_idx.tolist()
                    indices.extend(np.random.choice(split_idx, num_to_retain[g]).tolist())
            split_idx = np.sort(indices)
        
        else:
            if split not in self.split_dict:
                raise ValueError(f"Split {split} not found in dataset's split_dict.")
            split_mask = self.split_array == self.split_dict[split]
            if ordering is None:
                split_idx = np.where(split_mask)[0]
            else:
                if len(ordering) == torch.sum(split_mask):
                    pass
                else:
                    groups, group_counts = self._eval_grouper.metadata_to_group(
                        self.metadata_array[split_mask.ravel()],
                        return_counts=True)
                    assert len(group_counts) == len(ordering)
                    split_idx = np.where(split_mask)[0]
                    split_idx = np.cat(([split_idx[np.where(groups == g)[0]] for g in ordering]))
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = WILDSSubset(self, split_idx, transform)
        # if '_train' in split:
        subset.og_group_counts = og_group_counts
        return subset

    def check_init(self):
        """
        Convenience function to check that the WILDSDataset is properly configured.
        """
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'WILDSDataset is missing {attr_name}.'

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Check splits
        assert self.split_dict.keys()==self.split_names.keys()
        assert 'train' in self.split_dict
        assert 'val' in self.split_dict

        # Check the form of the required arrays
        assert (isinstance(self.y_array, torch.Tensor) or isinstance(self.y_array, list))
        assert isinstance(self.metadata_array, torch.Tensor), 'metadata_array must be a torch.Tensor'

        # Check that dimensions match
        assert len(self.y_array) == len(self.metadata_array)
        assert len(self.split_array) == len(self.metadata_array)

        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]

        # Check that it is not both classification and detection
        assert not (self.is_classification and self.is_detection)

        # For convenience, include y in metadata_fields if y_size == 1
        if self.y_size == 1:
            assert 'y' in self.metadata_fields

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset, e.g., 'amazon', 'camelyon17'.
        """
        return self._dataset_name
 
    @property
    def data_dir(self):
        """
        The full path to the folder in which the dataset is stored.
        """
        return self._data_dir

    @property
    def collate(self):
        """
        Torch function to collate items in a batch.
        By default returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)

    @property
    def split_scheme(self):
        """
        A string identifier of how the split is constructed,
        e.g., 'standard', 'mixed-to-test', 'user', etc.
        """
        return self._split_scheme

    @property
    def split_dict(self):
        """
        A dictionary mapping splits to integer identifiers (used in split_array),
        e.g., {'train': 0, 'val': 1, 'test': 2}.
        Keys should match up with split_names.
        """
        return getattr(self, '_split_dict', WILDSDataset.DEFAULT_SPLITS)

    @property
    def split_names(self):
        """
        A dictionary mapping splits to their pretty names,
        e.g., {'train': 'Train', 'val': 'Validation', 'test': 'Test'}.
        Keys should match up with split_dict.
        """
        return getattr(self, '_split_names', WILDSDataset.DEFAULT_SPLIT_NAMES)

    @property
    def split_array(self):
        """
        An array of integers, with split_array[i] representing what split the i-th data point
        belongs to.
        """
        return self._split_array

    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def y_size(self):
        """
        The number of dimensions/elements in the target, i.e., len(y_array[i]).
        For standard classification/regression tasks, y_size = 1.
        For multi-task or structured prediction settings, y_size > 1.
        Used for logging and to configure models to produce appropriately-sized output.
        """
        return self._y_size

    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)

    @property
    def is_classification(self):
        """
        Boolean. True if the task is classification, and false otherwise.
        """
        return getattr(self, '_is_classification', (self.n_classes is not None))

    @property
    def is_detection(self):
        """
        Boolean. True if the task is detection, and false otherwise.
        """
        return getattr(self, '_is_detection', False)

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array

    @property
    def metadata_map(self):
        """
        An optional dictionary that, for each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        For example, if we have
            metadata_fields = ['hospital', 'y']
            metadata_map = {'hospital': ['East', 'West']}
        then if metadata_array[i, 0] == 0, the i-th data point belongs to the 'East' hospital
        while if metadata_array[i, 0] == 1, it belongs to the 'West' hospital.
        """
        return getattr(self, '_metadata_map', None)

    @property
    def original_resolution(self):
        """
        Original image resolution for image datasets.
        """
        return getattr(self, '_original_resolution', None)

    def initialize_data_dir(self, root_dir):
        """
        Helper function for downloading/updating the dataset if required.
        Note that we only do a version check for datasets where the download_url is set.
        Currently, this includes all datasets except Yelp.
        Datasets for which we don't control the download, like Yelp,
        might not handle versions similarly.
        """
        os.makedirs(root_dir, exist_ok=True)
        data_dir = os.path.join(root_dir, f'{self.dataset_name}')
        print('data_dir ', data_dir)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'The {self.dataset_name} dataset could not be found in {data_dir}. {self.dataset_name} cannot be automatically downloaded. Please download it manually.')
        return data_dir

    @staticmethod
    def standard_eval(metric, y_pred, y_true):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results = {
            **metric.compute(y_pred, y_true),
        }
        results_str = (
            f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        )
        return results, results_str

    @staticmethod
    def standard_group_eval_old(metric, grouper, y_pred, y_true, metadata, aggregate=True):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - grouper (CombinatorialGrouper): Grouper object that converts metadata into groups
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results, results_str = {}, ''
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        g = grouper.metadata_to_group(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups)
        for group_idx in range(grouper.n_groups):
            group_str = grouper.group_field_str(group_idx)
            group_metric = group_results[metric.group_metric_field(group_idx)]
            group_counts = group_results[metric.group_count_field(group_idx)]
            results[f'{metric.name}_{group_str}'] = group_metric
            results[f'count_{group_str}'] = group_counts
            if group_results[metric.group_count_field(group_idx)] == 0:
                continue
            results_str += (
                f'  {grouper.group_str(group_idx)}  '
                f"[n = {group_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {group_results[metric.group_metric_field(group_idx)]:5.3f}\n")
        results[f'{metric.worst_group_metric_field}'] = group_results[f'{metric.worst_group_metric_field}']
        results_str += f"Worst-group {metric.name}: {group_results[metric.worst_group_metric_field]:.3f}\n"
        return results, results_str
    
    @staticmethod
    def standard_group_eval(metric, grouper, y_pred, y_true, metadata, aggregate=True):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - grouper (CombinatorialGrouper): Grouper object that converts metadata into groups
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results, results_str = {}, ''
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"

        g = grouper.metadata_to_group(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups)

        best_group_val = None            # ← track the best group score
        best_group_idx = None

        for group_idx in range(grouper.n_groups):
            group_str = grouper.group_field_str(group_idx)
            group_metric = group_results[metric.group_metric_field(group_idx)]
            group_counts = group_results[metric.group_count_field(group_idx)]

            results[f'{metric.name}_{group_str}'] = group_metric
            results[f'count_{group_str}'] = group_counts

            if group_counts == 0:
                continue

            # update best-group tracker
            if best_group_val is None or group_metric > best_group_val:
                best_group_val = group_metric
                best_group_idx = group_idx

            results_str += (
                f'  {grouper.group_str(group_idx)}  '
                f"[n = {group_counts:6.0f}]:\t"
                f"{metric.name} = {group_metric:5.3f}\n"
            )

        # worst group (already provided by Metric)
        worst_val = group_results[f'{metric.worst_group_metric_field}']
        results[f'{metric.worst_group_metric_field}'] = worst_val
        results_str += f"Worst-group {metric.name}: {worst_val:.3f}\n"

        # best group (new)
        if best_group_val is not None:
            best_group_key = f'best_{metric.name}'
            results[best_group_key] = best_group_val
            results_str += f"Best-group  {metric.name}: {best_group_val:.3f}\n"

        return results, results_str



class WILDSSubset(WILDSDataset):
    def __init__(self, dataset, indices, transform, do_transform_y=False):
        """
        This acts like `torch.utils.data.Subset`, but on `WILDSDatasets`.
        We pass in `transform` (which is used for data augmentation) explicitly
        because it can potentially vary on the training vs. test subsets.
        `do_transform_y` (bool): When this is false (the default),
                                 `self.transform ` acts only on  `x`.
                                 Set this to true if `self.transform` should
                                 operate on `(x,y)` instead of just `x`.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_scheme', '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.do_transform_y = do_transform_y

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y, metadata

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)
    
    def get_metadata_df(self, with_split=False):
        md = pd.DataFrame(self.metadata_array, columns=self.metadata_fields)
        if with_split:
            md['split'] = self._split_array
        return md

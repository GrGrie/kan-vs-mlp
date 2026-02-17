import os

import random
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.cm
import matplotlib.pyplot as plt

from itertools import product

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.model_selection import train_test_split
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont

def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr,
                                np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                arr,
                                np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr

class ColoredMNIST(WILDSDataset):
    _dataset_name = 'cmnist'

    def __init__(self, version=None, root_dir='', download=True, split_scheme='official',
                invar_str = 1.0, spur_str = 0.99, test_pct = 0.2, val_pct = 0.1, data_seed = 0): 
        self.invar_str = invar_str
        self.spur_str = spur_str
        self._data_dir = self.initialize_data_dir(root_dir)
        train_mnist = datasets.mnist.MNIST(self._data_dir, train=True, download=download)

        X, Y, G = [], [], []
        for idx, (im, label) in enumerate(train_mnist):
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with a% probability
            if np.random.uniform() < 1 - self.invar_str:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            if np.random.uniform() < 1 - self.spur_str:
                color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)
            binary_attr = int(not color_red)

            X.append(colored_arr)
            Y.append(binary_label)
            G.append(binary_attr)

        # Get the y values
        self._y_array = torch.LongTensor(Y)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(G), self._y_array),
            dim=1
        )
        self._metadata_fields = ['background', 'y']
        self._metadata_map = {
            'background': ['0', '1'], 
            'y': ['0', '1']
        }

        self.X = X
        self._original_resolution = (28, 28)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        test_idxs = np.random.choice(np.arange(len(train_mnist)), size = int(len(train_mnist) * test_pct), replace = False)
        val_idxs = np.random.choice(np.setdiff1d(np.arange(len(train_mnist)), test_idxs), size = int(len(train_mnist) * val_pct), replace = False)
        self._split_array = np.zeros((len(train_mnist), 1))
        self._split_array[val_idxs] = 1
        self._split_array[test_idxs] = 2
        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['background', 'y']))

        super().__init__(self._data_dir, split_scheme)


    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)

        results, results_str = self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

        return results, results_str

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       return Image.fromarray(self.X[idx])

class SpuriousCIFAR10(WILDSDataset):
    _dataset_name = 'spur_cifar10'

    def __init__(self, version=None, root_dir='', download=True, split_scheme='official',
                invar_str = 1., spur_str = 0.95, test_pct = 0.2, val_pct = 0.1, B = 0.5, data_seed = 0):
        self.invar_str = invar_str
        self.spur_str = spur_str
        self.B = B

        self._data_dir = self.initialize_data_dir(root_dir)

        cifar_train = datasets.cifar.CIFAR10(root = self._data_dir, train = True, download = download)
        cifar_test = datasets.cifar.CIFAR10(root = self._data_dir, train = False, download = download)

        train_X, train_Y = np.array(cifar_train.data), np.array(cifar_train.targets)
        test_X, test_Y = np.array(cifar_test.data), np.array(cifar_test.targets)

        classes = np.sort(np.unique(train_Y))
        dim = train_X.shape[1]

        configs = list(product([lambda x: 0.5 + 0.5*x, lambda x: 0.5 - 0.5 * x], repeat = 4))
        random.shuffle(configs)
        config_mapping = configs[:10]

        for ds in [train_Y, test_Y]:
            flip_inds = np.random.randint(0, len(ds), size = int(len(ds) * (1 - self.invar_str)))
            for cls in classes:
                cls_inds = np.intersect1d(flip_inds, (ds == cls).nonzero())
                ds[cls_inds] = np.random.choice(np.delete(classes, cls), size = len(cls_inds), replace = True)

        G = []
        spur_color_ids = []
        for X, Y in ((train_X, train_Y), (test_X, test_Y)):
            spu_config = np.random.random(len(X)) >= (1-self.spur_str)
            G.append(spu_config.astype(int))
            S_part = []
            if self.spur_str > 0.0:
                for i in range(len(X)):
                    y = Y[i]
                    if spu_config[i]:
                        spur_class = y
                    else:
                        spur_class = random.choice([cls for cls in range(10) if cls != y])

                    config = config_mapping[spur_class]

                    spur_color_ids.append(spur_class)
                    # config = config_mapping[y] if spu_config[i] else random.choice(config_mapping[:y] + config_mapping[y+1:])    
                    X[i, int(dim/2), : , 0] = config[0](self.B) # horizontal
                    # for ch in range(3):
                    #     X[i, :, int(dim/2), ch] = config[ch + 1](self.B) # vertical

        background_array = torch.from_numpy(np.concatenate(G)).long()
        spur_color_id_array = torch.tensor(spur_color_ids).long()
        self.X = np.concatenate((train_X, test_X))

        # Get the y values
        self._y_array = torch.from_numpy(np.concatenate((train_Y, test_Y))).long()
        self._y_size = 1
        self._n_classes = len(classes)

        self._metadata_array = torch.stack([
            background_array,
            self._y_array,
            spur_color_id_array
            ], dim=1)

        # self._metadata_array = torch.stack(
        #     (torch.from_numpy(np.concatenate(G)).long(), self._y_array),
        #     dim=1
        # )
        self._metadata_fields = ['background', 'y', 'spur_color_id']
        self._metadata_map = {
            'background': ['0', '1'], 
            'y': cifar_train.classes, 
            'spur_color_id': [str(i) for i in range(10)]
        }

        self._original_resolution = (32, 32)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        val_idxs = np.random.choice(np.arange(len(train_X)), size = int(len(train_X) * val_pct), replace = False)
        self._split_array = np.zeros((len(train_X) + len(test_X), 1))
        self._split_array[val_idxs] = 1
        self._split_array[len(train_X):] = 2

        # n_train = (self._split_array == 0).sum()
        # n_val   = (self._split_array == 1).sum()
        # n_test  = (self._split_array == 2).sum()
        # print(f"[INFO] Dataset sizes — Train: {n_train}, Val: {n_val}, Test: {n_test}")
        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['background', 'y']))

        super().__init__(self._data_dir, split_scheme)


    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)

        results, results_str = self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

        return results, results_str

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       return Image.fromarray(self.X[idx])

def overlay_logo(base_img, logo_img, pos=(0, 0)):
    base_pil = Image.fromarray(base_img).convert("RGBA")
    base_pil.paste(logo_img, pos, logo_img)
    return np.array(base_pil.convert("RGB"))

def make_background_transparent(path_in, path_out=None,
                                bg_color=(255, 255, 255),  # colour to make transparent
                                tol=5,                     # tolerance (0–255)
                                keep_alpha=False):
    """
    Turn `bg_color` pixels into transparent pixels.
    """
    img = Image.open(path_in).convert("RGBA")
    data = np.asarray(img).copy()
    
    r, g, b, a = np.rollaxis(data, axis=-1)
    mask = (abs(r - bg_color[0]) <= tol) & \
           (abs(g - bg_color[1]) <= tol) & \
           (abs(b - bg_color[2]) <= tol)
    data[..., 3][mask] = 0 if keep_alpha else 0  # fully transparent
    
    out = Image.fromarray(data, mode="RGBA")
    if path_out:
        out.save(path_out, "PNG")
    return out

def make_letter_logos(letter_a: str = "A",
                      letter_b: str = "B",
                      patch_px: int = 8,
                      font_px: int = 8,
                      color_a=(255, 255, 255),        # white
                      color_b=(255, 255, 255),        # white
                      alpha: int = 180,               # 0‑255
                      font_path: str | None = None):
    """
    Return (logo_a, logo_b) as RGBA PIL images, each patch_px×patch_px,
    with the letter placed at the *top‑left* of the patch.
    """

    def _single_letter(letter, rgb):
        img  = Image.new("RGBA", (patch_px, patch_px), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, font_px) if font_path else ImageFont.load_default()

        # Draw the glyph at (0,0) so it hugs the top‑left corner
        draw.text((0, 0), letter, font=font, fill=rgb + (alpha,))
        return img

    return (_single_letter(letter_a, color_a),
            _single_letter(letter_b, color_b))

class SpuriousCIFAR10_logo(WILDSDataset):
    _dataset_name = 'spur_cifar10'

    def __init__(self, version=None, root_dir='', download=True, split_scheme='official',
                invar_str=1., spur_str=0.95, test_pct=0.2, val_pct=0.1, B=0.5, data_seed=0):
        self.invar_str = invar_str
        self.spur_str = spur_str
        self.B = B

        self._data_dir = self.initialize_data_dir(root_dir)

        # Load CIFAR-10
        cifar_train = datasets.cifar.CIFAR10(root=self._data_dir, train=True, download=download)
        cifar_test = datasets.cifar.CIFAR10(root=self._data_dir, train=False, download=download)

        train_X, train_Y = np.array(cifar_train.data), np.array(cifar_train.targets)
        test_X, test_Y = np.array(cifar_test.data), np.array(cifar_test.targets)

        classes = np.sort(np.unique(train_Y))

        # Load logos
        # self.logo_a = Image.open('logo_a.png').convert("RGBA").resize((8, 8))
        # self.logo_b = Image.open('logo_b.png').convert("RGBA").resize((8, 8))
        # self.logo_a = make_background_transparent("logo_a.png").resize((8, 8), Image.ANTIALIAS)
        # self.logo_b = make_background_transparent("logo_b.png").resize((8, 8), Image.ANTIALIAS)
        self.logo_a, self.logo_b = make_letter_logos(
            letter_a="A",
            letter_b="B",
            patch_px=8,
            font_px=8,
            alpha=180
        )

        # Invariant label corruption
        for ds in [train_Y, test_Y]:
            flip_inds = np.random.randint(0, len(ds), size=int(len(ds) * (1 - self.invar_str)))
            for cls in classes:
                cls_inds = np.intersect1d(flip_inds, (ds == cls).nonzero())
                ds[cls_inds] = np.random.choice(np.delete(classes, cls), size=len(cls_inds), replace=True)

        # Apply spurious logo overlay
        G = []

        for X, Y in ((train_X, train_Y), (test_X, test_Y)):
            spur_indicator = np.zeros(len(X), dtype=int)
            for cls in classes:
                cls_inds = np.where(Y == cls)[0]
                n_spur = int(len(cls_inds) * self.spur_str)

                np.random.seed(data_seed + cls)
                selected = np.random.choice(cls_inds, size=n_spur, replace=False)

                for i in cls_inds:
                    logo = self.logo_a if i in selected else self.logo_b
                    X[i] = self._overlay_logo(X[i], self.logo_a, pos=(0, 0))
                    spur_indicator[i] = 1 if i in selected else 0

            G.append(spur_indicator)

        self.X = np.concatenate((train_X, test_X))

        # Get the y values
        self._y_array = torch.from_numpy(np.concatenate((train_Y, test_Y))).long()
        self._y_size = 1
        self._n_classes = len(classes)

        self._metadata_array = torch.stack(
            (torch.from_numpy(np.concatenate(G)).long(), self._y_array),
            dim=1
        )
        self._metadata_fields = ['background', 'y']
        self._metadata_map = {
            'background': ['0', '1'],
            'y': cifar_train.classes
        }

        self._original_resolution = (32, 32)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        val_idxs = np.random.choice(np.arange(len(train_X)), size=int(len(train_X) * val_pct), replace=False)
        self._split_array = np.zeros((len(train_X) + len(test_X), 1))
        self._split_array[val_idxs] = 1
        self._split_array[len(train_X):] = 2

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['background', 'y']))

        super().__init__(self._data_dir, split_scheme)

    def _overlay_logo(self, base_img: np.ndarray,
                  logo_img: Image.Image,
                  pos=(0, 0)):               # ← default is now top‑left
        canvas = Image.fromarray(base_img).convert("RGBA")
        canvas.paste(logo_img, pos, logo_img)    # logo supplies its own alpha mask
        return np.array(canvas.convert("RGB"))

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        results, results_str = self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
        return results, results_str

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        return Image.fromarray(self.X[idx])

class MixedSpuriousCIFAR10(WILDSDataset):
    _dataset_name = 'spur_cifar10'

    def __init__(self, version=None, root_dir='', download=True, split_scheme='official',
                 spur_frac=0.1, spur_str=0.95, invar_str=1.0, test_pct=0.2, val_pct=0.1, B=0.5, data_seed=0):

        self.spur_str = spur_str
        self.spur_frac = spur_frac
        self.invar_str = invar_str
        self.B = B

        self._data_dir = self.initialize_data_dir(root_dir)

        # Load clean CIFAR-10
        cifar_clean = datasets.cifar.CIFAR10(root=self._data_dir, train=True, download=download)
        train_X_clean, train_Y_clean = np.array(cifar_clean.data), np.array(cifar_clean.targets)

        # Load spurious CIFAR-10 logic
        cifar_spur = datasets.cifar.CIFAR10(root=self._data_dir, train=True, download=download)
        train_X_spur, train_Y_spur = np.array(cifar_spur.data), np.array(cifar_spur.targets)

        classes = np.sort(np.unique(train_Y_clean))
        dim = train_X_clean.shape[1]
        configs = list(product([lambda x: 0.5 + 0.5 * x, lambda x: 0.5 - 0.5 * x], repeat=4))
        random.seed(data_seed)
        random.shuffle(configs)
        config_mapping = configs[:10]

        # Label flipping (invariant corruption) — only applied to spurious subset
        flip_inds = np.random.randint(0, len(train_Y_spur), size=int(len(train_Y_spur) * (1 - invar_str)))
        for cls in classes:
            cls_inds = np.intersect1d(flip_inds, (train_Y_spur == cls).nonzero())
            train_Y_spur[cls_inds] = np.random.choice(np.delete(classes, cls), size=len(cls_inds), replace=True)

        # Spurious signal injection (only for selected portion)
        G_clean = np.zeros(len(train_X_clean), dtype=int)  # All clean samples marked 0
        G_spur = np.zeros(len(train_X_spur), dtype=int)

        for i in range(len(train_X_spur)):
            y = train_Y_spur[i]
            apply_spur = np.random.uniform() < spur_str
            G_spur[i] = 1 if apply_spur else 0
            config = config_mapping[y] if apply_spur else random.choice(config_mapping[:y] + config_mapping[y + 1:])
            train_X_spur[i, int(dim / 2), :, 0] = config[0](B)

        # Sample subsets
        rng = np.random.default_rng(data_seed)
        n_total = len(train_X_clean)
        n_spur = int(spur_frac * n_total)
        n_clean = n_total - n_spur

        spur_indices = rng.choice(len(train_X_spur), size=n_spur, replace=False)
        clean_indices = rng.choice(len(train_X_clean), size=n_clean, replace=False)

        X = np.concatenate((train_X_clean[clean_indices], train_X_spur[spur_indices]))
        Y = np.concatenate((train_Y_clean[clean_indices], train_Y_spur[spur_indices]))
        G = np.concatenate((G_clean[clean_indices], G_spur[spur_indices]))

        # Shuffle combined dataset
        shuffled_idx = np.random.permutation(len(X))
        self.X = X[shuffled_idx]
        Y = Y[shuffled_idx]
        G = G[shuffled_idx]

        # Metadata + labels
        self._y_array = torch.from_numpy(Y).long()
        self._y_size = 1
        self._n_classes = 10

        self._metadata_array = torch.stack([
            torch.from_numpy(G).long(),
            self._y_array
        ], dim=1)
        self._metadata_fields = ['background', 'y']
        self._metadata_map = {
            'background': ['0', '1'],
            'y': cifar_clean.classes
        }

        self._original_resolution = (32, 32)

        # Split array
        self._split_scheme = split_scheme
        if split_scheme != 'official':
            raise ValueError(f"Split scheme {split_scheme} not recognized")

        val_idxs = np.random.choice(np.arange(len(X)), size=int(val_pct * len(X)), replace=False)
        test_idxs = np.random.choice(np.setdiff1d(np.arange(len(X)), val_idxs), size=int(test_pct * len(X)), replace=False)
        self._split_array = torch.zeros((len(X), 1), dtype=torch.long)
        self._split_array[val_idxs] = 1
        self._split_array[test_idxs] = 2

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(['background', 'y']))
        super().__init__(self._data_dir, split_scheme)

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        results, results_str = self.standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)
        return results, results_str

    def get_input(self, idx):
        return Image.fromarray(self.X[idx])


# new dataset class to store features
class FeatureDataset(Dataset):
    def __init__(self, features, labels, metadata):
        """
        Args:
            features (torch.Tensor): Extracted features from the encoder.
            labels (torch.Tensor): Corresponding labels.
            metadata (torch.Tensor): Metadata information.
        """
        self.features = features
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.metadata[idx]

# new dataset class to store features
class FeatureDataset_Cifar10(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (torch.Tensor): Extracted features from the encoder.
            labels (torch.Tensor): Corresponding labels.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def safe_random_choice(exclude_idx, max_idx):
    indices = list(range(max_idx))
    indices.remove(exclude_idx)
    return random.choice(indices)
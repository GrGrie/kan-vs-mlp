import wilds

def get_dataset(dataset, **dataset_kwargs):
    """
    Returns the appropriate WILDS dataset class.
    Input:
        dataset (str): Name of the dataset
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified WILDSDataset class.
    """

    if dataset not in wilds.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {wilds.supported_datasets}.')

    elif dataset == 'celebA':
        from wilds.datasets.celebA_dataset import CelebADataset
        return CelebADataset(**dataset_kwargs)

    elif dataset == 'waterbirds':
        from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
        return WaterbirdsDataset(**dataset_kwargs)

    elif dataset == 'cmnist':
        from wilds.datasets.synthetic import ColoredMNIST
        return ColoredMNIST(**dataset_kwargs)

    elif dataset == 'spur_cifar10':
        from wilds.datasets.synthetic import SpuriousCIFAR10
        return SpuriousCIFAR10(**dataset_kwargs)

    elif dataset == 'mixed_spur_cifar10':
        from wilds.datasets.synthetic import MixedSpuriousCIFAR10
        return MixedSpuriousCIFAR10(**dataset_kwargs)
    
    elif dataset == 'cifar10':
        from wilds.datasets.nospur_datasets import WILDSCIFAR10
        return WILDSCIFAR10(**dataset_kwargs)

    elif dataset == 'metashift':
        from wilds.datasets.metashift_dataset import MetaShiftCatsDogsDataset
        return MetaShiftCatsDogsDataset(**dataset_kwargs)

    elif dataset == 'stl10': 
        from wilds.datasets.stl10_dataset import STL10Dataset
        return STL10Dataset(**dataset_kwargs)

    elif dataset == 'iwildcam': 
        from wilds.datasets.iWildCam_dataset import IWildCamDataset
        return IWildCamDataset(**dataset_kwargs)
    
    elif dataset == 'hard_imagenet':
        from wilds.datasets.hard_imagenet_dataset import HardImageNet
        return HardImageNet(**dataset_kwargs)

    elif dataset == 'bgchallenge':
        from wilds.datasets.bgchallenge_dataset import ImageNetBGDataset
        return ImageNetBGDataset(**dataset_kwargs)

    elif dataset == 'cxr':
        from wilds.datasets.cxr_dataset import CXRDataset
        return CXRDataset(**dataset_kwargs)
    
    elif dataset == 'nicopp':
        from wilds.datasets.nicopp_dataset import NICOppDataset
        return NICOppDataset(**dataset_kwargs)

import torch
from torch.utils.data import DataLoader, Dataset
import loaders.datasets as datasets


def get_dataloader(args, split, shuffle=True, out_name=False, sample=None, selection=None, mode=None, seed=None):


    """
    Creates and returns a DataLoader for the specified dataset and configuration.

    Args:
        args (Namespace): Configuration arguments containing dataset and loader parameters.
        split (str): The dataset split to use (e.g., 'train', 'val', 'test').
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
        out_name (bool, optional): Whether to output the name of the dataset. Default is False.
        sample (tuple, optional): A tuple specifying the sampling strategy (iter, way, shot, query). Default is None.
        selection (list, optional): A list of class selections for the dataset. Default is None.
        mode (str, optional): The mode of the dataset (e.g., 'train', 'val', 'test'). Default is None.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        DataLoader: A PyTorch DataLoader configured with the specified dataset and parameters.
    """
    # sample: iter, way, shot, query
    if args.fsl:
        ts_condition = split
    else:
        ts_condition = mode
    transform = datasets.make_transform(args, ts_condition)
    sets = datasets.DatasetFolder(args.data_root, args.dataset, split, transform, out_name=out_name, cls_selction=selection, mode=mode)
    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample, seed)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.num_workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.num_workers, pin_memory=True)
    return loader

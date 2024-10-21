import numpy as np
import torch
from torch.utils.data import Sampler
import torch

__all__ = ['CategoriesSampler']


class CategoriesSampler(Sampler):
    """
    A PyTorch Sampler for episodic training in few-shot learning tasks.

    Args:
        label (list or numpy.ndarray): Array of labels corresponding to the dataset.
        n_iter (int): Number of iterations (episodes) per epoch.
        n_way (int): Number of classes per episode.
        n_shot (int): Number of samples per class in the support set.
        n_query (int): Number of samples per class in the query set.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Attributes:
        n_iter (int): Number of iterations (episodes) per epoch.
        n_way (int): Number of classes per episode.
        n_shot (int): Number of samples per class in the support set.
        n_query (int): Number of samples per class in the query set.
        seed (int, optional): Random seed for reproducibility.
        m_ind (list): List of tensors, each containing indices of samples for a specific class.

    Methods:
        __len__(): Returns the number of iterations (episodes) per epoch.
        __iter__(): Yields batches of indices for each episode.
    """

    def __init__(self, label, n_iter, n_way, n_shot, n_query, seed=None):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.seed = seed

        label = np.array(label)
        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            if self.seed:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            if self.seed:
                print(classes)
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch_gallery.append(l[pos[:self.n_shot]])
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
            batch = torch.cat(batch_gallery + batch_query)
            yield batch

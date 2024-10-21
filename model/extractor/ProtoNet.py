import torch.nn as nn
import torch
import torch.nn.functional as F


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


class ProtoNet(nn.Module):
    """
    ProtoNet is a neural network module for prototypical networks used in few-shot learning tasks.

    Attributes:
        encoder (nn.Module): The feature extraction network.
        train_info (list): Training configuration parameters including the number of ways, shots, queries, and the metric name.

    Methods:
        __init__(feature_net, args=None):
            Initializes the ProtoNet with a feature extraction network and optional training arguments.
            
        forward(data, _=False):
            Forward pass of the network. If in training mode, computes the logits based on the prototypical network approach.
            If not in training mode, simply returns the encoded features.
            
            Args:
                data (Tensor): Input data to the network.
                _ (bool, optional): Unused argument, defaults to False.
            
            Returns:
                Tensor: Logits if in training mode, otherwise encoded features.
    """

    def __init__(self, feature_net, args=None):
        super().__init__()
        self.encoder = feature_net
        if args is None:
            self.train_info = [30, 1, 15, 'euclidean']
            # self.val_info = [5, 1, 15]
        else:
            self.train_info = [args.meta_train_way, args.meta_train_shot, args.meta_train_query, args.meta_train_metric]
            # self.val_info = [args.meta_val_way, args.meta_val_shot, args.meta_val_query]

    def forward(self, data, _=False):
        if not self.training:
            return self.encoder(data, True)
        way, shot, query, metric_name = self.train_info
        proto, _ = self.encoder(data, True)
        shot_proto, query_proto = proto[:shot * way], proto[shot * way:]
        shot_proto = shot_proto.reshape(way, shot, -1).mean(1)
        logits = -get_metric(metric_name)(shot_proto, query_proto)

        return logits

from args_setting import *
from torch.utils.data import DataLoader, Dataset
import matplotlib.cm as mpl_color_map
import copy
from tqdm.auto import tqdm
import tools.calculate_tool as cal
from model.FSL import FSLSimilarity, SimilarityLoss
from loaders.base_loader import get_dataloader
from PIL import Image
import numpy as np
import torch


@torch.no_grad()
"""
Evaluates the performance of a given model on a dataset.

Args:
    model (torch.nn.Module): The model to evaluate.
    data_loader (torch.utils.data.DataLoader): DataLoader providing the dataset.
    device (torch.device): The device to run the evaluation on (e.g., 'cpu' or 'cuda').
    criterion (callable): The loss function and accuracy metric to evaluate the model.

Returns:
    None

This function performs the following steps:
1. Sets the model and criterion to evaluation mode.
2. Iterates over the data_loader to get batches of inputs and targets.
3. Moves inputs to the specified device and performs a forward pass through the model.
4. Computes the loss and accuracy using the criterion.
5. Accumulates the loss and accuracy metrics.
6. Prints the average loss, accuracy, and confidence interval of the accuracy.

Note:
    This function uses `torch.no_grad()` to disable gradient computation during evaluation.
"""
def evaluate(model, data_loader, device, criterion):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the dataset to evaluate.
        device (torch.device): The device (CPU or GPU) to perform the evaluation on.
        criterion (callable): The loss function or criterion used to compute the loss and accuracy.

    Returns:
        None

    Prints:
        The average loss, accuracy at 95% confidence interval, and the confidence interval itself.
    """
    model.eval()
    criterion.eval()
    print("start test: ")
    running_loss = 0.0
    running_att_loss = 0.0
    running_acc_95 = []
    L = len(data_loader)
    for i, (inputs, target) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device, dtype=torch.float32)
        out, att_loss = model(inputs)
        loss, acc, logits = criterion(out, att_loss)
        a = loss.item()
        running_loss += a
        running_att_loss += att_loss.item()
        running_acc_95.append(round(acc.item(), 4))

    print("loss: ", round(running_loss/L, 3))
    print("acc_95: ", round(cal.compute_confidence_interval(running_acc_95)[0], 4))
    print("interval: ", round(cal.compute_confidence_interval(running_acc_95)[1], 4))


def main(name):
    criterien = SimilarityLoss(args)
    model = FSLSimilarity(args)
    model_name = name
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    sample_info_val = [args.test_episodes, args.n_way, args.n_shot, args.query]
    loaders_test = get_dataloader(args, "test", sample=sample_info_val)
    evaluate(model, loaders_test, device, criterien)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    args.slot_base_train = False
    args.double = False
    model_name = (f"{args.dataset}_{args.base_model}_slot{args.num_slot}_" + 'fsl_checkpoint.pth')
    main(model_name)
from args_setting import *
from model.model_tools import print_param
from model.FSL import FSLSimilarity, SimilarityLoss
import torch
import time
import datetime
import tools.prepare_things as prt
from pathlib import Path
from engine_fsl import train_one_epoch, evaluate
from tools.calculate_tool import MetricLogSimilar
from tools.Adabelif import AdaBelief
from loaders.base_loader import get_dataloader
import numpy as np


def main(args):
    """
    Main function to train and evaluate the FSL (Few-Shot Learning) model.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - device (str): The device to use for training (e.g., 'cuda' or 'cpu').
            - train_episodes (int): Number of training episodes.
            - n_way (int): Number of classes in each episode.
            - n_shot (int): Number of support samples per class.
            - query (int): Number of query samples per class.
            - val_episodes (int): Number of validation episodes.
            - dataset (str): Name of the dataset.
            - base_model (str): Name of the base model.
            - num_slot (int): Number of slots to use.
            - output_dir (str): Directory to save the output.
            - lr (float): Learning rate.
            - lr_drop (int): Step size for learning rate decay.
            - start_epoch (int): Starting epoch number.
            - epochs (int): Total number of epochs.

    Returns:
        None
    """
    device = torch.device(args.device)
    sample_info_train = [args.train_episodes, args.n_way, args.n_shot, args.query]
    loaders_train = get_dataloader(args, "train", sample=sample_info_train)
    sample_info_val = [args.val_episodes, args.n_way, args.n_shot, args.query]
    loaders_val = get_dataloader(args, "val", sample=sample_info_val)
    criterien = SimilarityLoss(args).to(device)
    model = FSLSimilarity(args)

    model_name = f"{args.dataset}_{args.base_model}_use_slot_{args.num_slot}checkpoint.pth"
    model.to(device)
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"], strict=False)
    print("load pre-model " + model_name + " ready")

    print_param(model)
    params = [p for p in model.parameters() if p.requires_grad]

    output_dir = Path(args.output_dir)
    # optimizer = torch.optim.AdamW(params, lr=args.lr)
    optimizer = AdaBelief(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    print("Start training")
    start_time = time.time()
    log = MetricLogSimilar(args)
    record = log.record

    max_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, loaders_train, device, record, epoch, optimizer, criterien)
        evaluate(model, loaders_val, device, record, epoch, criterien)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / model_name]
            if record["val"]["accm"][epoch-1] > max_acc:
                print("get higher acc save current model")
                max_acc = record["val"]["accm"][epoch-1]
                for checkpoint_path in checkpoint_paths:
                    prt.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
        log.print_metric()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.slot_base_train = False
    args.double = False
    args.fsl = True
    if args.random:
        selection = np.random.randint(0, args.num_classes, args.num_slot)
    else:
        selection = np.arange(0, args.num_classes, args.interval)
    print(selection)
    args.num_slot = len(selection)
    model_name = (f"{args.dataset}_{args.base_model}_slot{args.num_slot}_" + 'fsl_checkpoint.pth')
    print("patterns num: ", args.num_slot)
    main(args)
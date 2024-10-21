import argparse


def get_args_parser():
    """
    Creates an argument parser for the FSL Project.

    Returns:
        argparse.ArgumentParser: The argument parser with predefined arguments.

    Arguments:
        --dataset (str): Name of the dataset. Default is "miniImageNet".
        --data_root (str): Root directory of the dataset. Default is "/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e/FSL_data/".

        --base_model (str): Base model architecture. Default is 'resnet18'.
        --channel (int): Number of channels. Default is 512.
        --num_classes (int): Number of classes. Default is 64.

        --n_way (int): Number of ways for FSL. Default is 5.
        --n_shot (int): Number of shots for FSL. Default is 1.
        --query (int): Number of query samples. Default is 15.

        --train_episodes (int): Number of training episodes. Default is 500.
        --val_episodes (int): Number of validation episodes. Default is 2000.
        --test_episodes (int): Number of test episodes. Default is 2000.

        --fsl (bool): Whether to use FSL model. Default is True.
        --lr (float): Learning rate. Default is 0.001.
        --lr_drop (int): Learning rate drop interval. Default is 10.
        --batch_size (int): Batch size. Default is 256.
        --weight_decay (float): Weight decay. Default is 0.0001.
        --epochs (int): Number of epochs. Default is 20.
        --img_size (int): Image size. Default is 80.
        --aug (bool): Whether to use augmentation. Default is True.
        --use_slot (bool): Whether to use slot module. Default is True.
        --fix_parameter (bool): Whether to fix parameters for backbone. Default is True.
        --double (bool): Whether to use double mode. Default is False.

        --num_slot (int): Number of slots. Default is 7.
        --interval (int): Interval for category sampling. Default is 10.
        --drop_dim (bool): Whether to drop dimension for average. Default is False.
        --slot_base_train (bool): Whether to use slot base training. Default is True.
        --use_pre (bool): Whether to use pre-trained parameters for backbone. Default is True.
        --loss_status (int): Status of loss (positive or negative). Default is 1.
        --hidden_dim (int): Dimension of to_k. Default is 64.
        --slots_per_class (int): Number of slots per class. Default is 1.
        --power (float): Power of the slot loss. Default is 2.
        --to_k_layer (int): Number of layers in to_k. Default is 3.
        --lambda_value (str): Lambda value of slot loss. Default is "1.".
        --vis (bool): Whether to save slot visualization. Default is False.
        --vis_id (int): ID of the image to visualize. Default is 0.
        --DT (bool): Whether to use DT training. Default is True.
        --random (bool): Whether to randomly select category. Default is False.

        --device (str): Device to use for training/testing. Default is 'cuda'.
        --output_dir (str): Directory to save outputs. Default is 'saved_model'.
        --start_epoch (int): Starting epoch. Default is 0.
        --num_workers (int): Number of workers. Default is 0.

        --world_size (int): Number of distributed processes. Default is 1.
        --local_rank (int): Local rank for distributed training.
        --dist_url (str): URL for setting up distributed training. Default is 'env://'.
    """
    parser = argparse.ArgumentParser('FSL Project', add_help=False)

    # dataset setting
    parser.add_argument('--dataset', default="miniImageNet", type=str)
    parser.add_argument('--data_root', default="/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e/FSL_data/", type=str)

    # model setting
    parser.add_argument('--base_model', default='resnet18', type=str)
    parser.add_argument('--channel', default=512, type=int)
    parser.add_argument("--num_classes", default=64, type=int)

    # FSL setting
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--query', default=15, type=int)

    parser.add_argument('--train_episodes', default=500, type=int)
    parser.add_argument('--val_episodes', default=2000, type=int)
    parser.add_argument('--test_episodes', default=2000, type=int)

    # train setting
    parser.add_argument('--fsl', default=True, type=bool, help='whether fsl model')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--img_size', default=80, help='path for save data')
    parser.add_argument('--aug', default=True, help='whether use augmentation')
    parser.add_argument('--use_slot', default=True, type=bool, help='whether use slot module')
    parser.add_argument('--fix_parameter', default=True, type=bool, help='whether fix parameter for backbone')
    parser.add_argument('--double', default=False, type=bool, help='whether double mode')

    # slot setting
    parser.add_argument('--num_slot', default=7, type=int, help='number of slot')
    parser.add_argument('--interval', default=10, type=int, help='skip applied in category sampling')
    parser.add_argument('--drop_dim', default=False, type=bool, help='drop dim for avg')
    parser.add_argument('--slot_base_train', default=True, type=bool, help='drop dim for avg')
    parser.add_argument('--use_pre', default=True, type=bool, help='whether use pre parameter for backbone')
    parser.add_argument('--loss_status', default=1, type=int, help='positive or negative loss')
    parser.add_argument('--hidden_dim', default=64, type=int, help='dimension of to_k')
    parser.add_argument('--slots_per_class', default=1, type=int, help='number of slot for each class')
    parser.add_argument('--power', default=2, type=float, help='power of the slot loss')
    parser.add_argument('--to_k_layer', default=3, type=int, help='number of layers in to_k')
    parser.add_argument('--lambda_value', default="1.", type=str, help='lambda  of slot loss')
    parser.add_argument('--vis', default=False, type=bool, help='whether save slot visualization')
    parser.add_argument('--vis_id', default=0, type=int, help='choose image to visualization')
    parser.add_argument('--DT', default=True, type=bool, help='DT training')
    parser.add_argument('--random', default=False, type=bool, help='whether random select category')

    # data/machine set
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='saved_model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser
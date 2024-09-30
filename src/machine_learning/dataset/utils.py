import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from utils.utils_logs import *

# Code modified from the Flower project on GitHub
# Repository link: https://github.com/adap/flower
def prepare_dataset(trainset, testset, num_partitions: int, batch_size_client: int, 
                    batch_size_test: int, val_ratio: float = 0.1):
    """
    This function partitions the training set into N disjoint
    subsets, each will become the local dataset of a client. This
    function also subsequently partitions each traininset partition
    into train and validation. The test set is left intact and will
    be used by the central server to asses the performance of the
    global model.
    """

    log_info(f"Trainset: {len(trainset)} samples, Testset: {len(testset)} samples.")

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions       # [num_images, num_images...]
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )
    log_info(f"Set of each client has {len(trainsets[0])} samples. Evaluation: {val_ratio*100} %")

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        # split trainset into training and valuating sets
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size_client, shuffle=True, num_workers=1)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size_client, shuffle=False, num_workers=1)
        )

    # create dataloader for the test set
    testloader = DataLoader(testset, batch_size=batch_size_test)
    return trainloaders, valloaders, testloader
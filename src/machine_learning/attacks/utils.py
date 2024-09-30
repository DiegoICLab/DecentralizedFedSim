import random
import numpy as np
from typing import List, Tuple

from torch.utils.data import DataLoader
from functools import reduce


def client_label_flipping_attack(
    dataloader: DataLoader, 
    percentage_flip=0.3
    ) -> DataLoader:
    """
    This function performs a label-flipping attack on the given dataloader.

    Args:
        dataloader: The dataloader to be attacked.
        percentage_flip: The percentage of labels to flip.

    Returns:
        att_dataloader: The modified dataloader.
    """
    # TODO Check 
    num_labels = 10
    att_dataloader = dataloader

    # Determine the number of samples to flip labels
    num_samples_flip = int(len(att_dataloader.dataset) * percentage_flip)

    # Get random indices of samples to flip labels
    indexs = [att_dataloader.dataset.dataset.indices[i] for i in att_dataloader.dataset.indices]
    flip_indices = random.sample(indexs, num_samples_flip)

    # Flip labels for selected samples
    for index in flip_indices:
        _, label = att_dataloader.dataset.dataset.dataset[index]
        new_label = (num_labels - label - 1) % num_labels 
        att_dataloader.dataset.dataset.dataset.targets[index] = new_label

    return att_dataloader

def client_sign_flipping_attack(model):
    """
    This function performs a sign-flipping attack on the given model.

    Args:
        model: the parameter model to be modified

    Returns:
        flipped_model: the modified model
    """

    flipped_model = [ -1 * param for param in model]
    return flipped_model


def client_noise_attack(model):
    """
    This function performs a noise attack on the given model.

    Args:
        model: the parameter model to be modified

    Returns:
        model: the modified model
    """

    for i in range(len(model)):
        model[i] += generate_gaussian_samples(0.1, 0.1, 1)

    return model

def client_IPM_attack(models, epsilon):
    """
    This function performs a IPM attack on the given models.

    Args:
        models: the parameter models employed to generate the malicious attack
        epsilon: configuration parameter
    Returns:
        IPM_model: the computed model
    """

    num_benign = len(models)
    IPM_model = ((-1) * epsilon / num_benign) * np.sum(models, axis = 0)
    return IPM_model

def client_ALIE_attack(models, zmax):
    """
    This function performs an ALIE attack on the given models.

    Args:
        models: the parameter models employed to generate the malicious attack
        zmax: configuration parameter
    Returns:
        ALIE_attack: the computed model
    """
    means = np.mean(models, axis=0)
    stds = np.std(models, axis=0)

    ALIE_attack = []
    for mean, std in zip(means, stds):
        sample = np.random.uniform(mean - zmax * std, mean + zmax * std)
        ALIE_attack.append(sample)
    return ALIE_attack


#### OTHER
def generate_gaussian_samples(mean, variance, num_samples):
    return np.random.normal(mean, np.sqrt(variance), num_samples)
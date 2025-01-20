from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_MNIST(data_path: str = "./data"):
    """
    This function downloads the MNIST dataset into the `data_path`
    directory if it is not there already. WE construct the train/test
    split by converting the images into tensors and normalising them
    """

    # transformation to convert images to tensors and apply normalisation
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # prepare train and test set
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


# #############################################################################
# Data analysis and centralized execution
# #############################################################################
def visualise_n_random_examples(trainset_, n: int, verbose: bool = True):
    # take n examples at random
    idx = list(range(len(trainset_.data)))
    random.shuffle(idx)
    idx = idx[:n]
    if verbose:
        print(f"[INFO] Will display images with idx: {idx}")

    # construct canvas
    num_cols = 8
    num_rows = int(np.ceil(len(idx) / num_cols))
    fig, axs = plt.subplots(figsize=(16, num_rows * 2), nrows=num_rows, ncols=num_cols)

    # display images on canvas
    for c_i, i in enumerate(idx):
        axs.flat[c_i].imshow(trainset_.data[i], cmap="gray")


def data_analysis():
    trainset, testset = load_MNIST()
    print(trainset)
    print("Labels: " + str(trainset.targets))
    print("Number of samples: " + str(len(trainset)))

    # construct histogram
    all_labels = trainset.targets
    num_possible_labels = len(set(all_labels.numpy().tolist()))  # this counts unique labels (so it should be = 10)
    plt.clf()
    plt.hist(all_labels, bins=num_possible_labels)
    plt.savefig("./output/MNIST/histogram.png")

    # plot formatting
    plt.clf()
    plt.xticks(range(num_possible_labels))
    plt.grid()
    plt.xlabel("Label")
    plt.ylabel("Number of images")
    plt.title("Class labels distribution for MNIST")

    # it is likely that the plot this function will generate looks familiar to other plots you might have generated before
    # or you might have encountered in other tutorials. So far, we aren't doing anything new, Federated Learning will start soon!
    visualise_n_random_examples(trainset, n=32)
    plt.savefig("./output/MNIST/data_analysis.png")

# ################################################################################
# Given the loaders.dataset, it is represented the distribution of labels (MNIST)
# ################################################################################
def plot_labels_distribution(train_partition):
    # count data points
    partition_indices = train_partition.indices
    print(f"[INFO] Number of images for training of each client: {len(partition_indices)}")

    # visualise histogram
    sns.set(style="whitegrid")
    labels, counts = np.unique(train_partition.dataset.dataset.targets[partition_indices], return_counts=True)
    plt.clf()
    plt.bar(labels, counts, color=sns.color_palette("pastel"))
    plt.xlabel("Label")
    plt.xticks(range(10))
    plt.ylabel("Number of images")
    plt.title("Class labels distribution for MNIST")
    # for i, count in enumerate(counts):
    #     plt.text(labels[i], count + 0.1, str(labels[i]), ha='center')
    plt.savefig("./output/simulation/simulation.png")


def model_analysis():
    model = MNIST_Net(num_classes=10)
    num_parameters = sum(value.numel() for value in model.state_dict().values())
    print(f"{num_parameters = }")


def run_centralised(epochs: int, lr: float, momentum: float = 0.9):
    """A minimal (but complete) training loop"""

    # instantiate the model
    model = MNIST_Net(num_classes=10)

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # get dataset and construct a dataloaders
    trainset, testset = load_MNIST()
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128)

    # train for the specified number of epochs
    model.train_model(trainloader, optim, epochs)

    # training is completed, then evaluate model on the test set
    loss, accuracy = model.test_model(trained_model, testloader)
    print(f"{loss = }")
    print(f"{accuracy = }")


def main():
    print("[INFO] Data analysis of dataset")
    data_analysis()
    print("[INFO] Analysis of model")
    model_analysis()
    print("[INFO] Running centralized training model and evaluation")
    run_centralised(epochs=5, lr=0.01)

if __name__ == "__main__":
    main()
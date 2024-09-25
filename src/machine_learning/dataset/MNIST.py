import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from tqdm import tqdm

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader (MNIST)
# #############################################################################

class MNIST_Net(nn.Module):
    def __init__(self, num_classes: int) -> None:       # 44,426 parameters with 10 classes
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_MNIST(net, trainloader, epochs, DEVICE):
    """Train the network on the training set."""
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    # TODO inputs, targets = self.trainset
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            if DEVICE is not None:
                loss_fn(net(images.to(DEVICE)), labels.to(DEVICE)).backward()         # Calcula la pérdida entre las predicciones del modelo y las etiquetas reales, y luego propaga hacia atrás los gradientes a través de la red neuronal.
            else:
                loss_fn(net(images), labels).backward()         # Calcula la pérdida entre las predicciones del modelo y las etiquetas reales, y luego propaga hacia atrás los gradientes a través de la red neuronal.
            optimizer.step()


def test_MNIST(net, testloader, DEVICE):
    """Validate the network on the entire test set."""
    loss_fn = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            if DEVICE is not None:
                outputs = net(images.to(DEVICE))
                labels = labels.to(DEVICE)
            else:
                outputs = net(images)
            loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()                       # Predicciones correctas en el lote. torch.max(·)[1] para obtener las clases predichas por el modelo.m()
    accuracy = correct / len(testloader.dataset)
    average_loss = loss / len(testloader.dataset)
    return accuracy, average_loss

def load_MNIST(data_path: str = "./data"):
    """This function downloads the MNIST dataset into the `data_path`
    directory if it is not there already. WE construct the train/test
    split by converting the images into tensors and normalising them"""

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
# REVISAR PORQUE ES MUY PARECIDO AL DE ARRIBA
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
    trained_model = train_MNIST(model, trainloader, optim, epochs)

    # training is completed, then evaluate model on the test set
    loss, accuracy = test_MNIST(trained_model, testloader)
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
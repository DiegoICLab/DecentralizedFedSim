import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from collections import OrderedDict
from tqdm import tqdm

# Code modified from the Flower project on GitHub
# Repository link: https://github.com/adap/flower

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader (CIFAR10)
# #############################################################################
class CIFAR10_Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: 
    A 60 Minute Blitz')"""
    
    def __init__(self, num_classes: int) -> None:   # 62006 parameters with 10 classes
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)           # Convolutional layer with 3 input channels, 6 output channels, and a 5x5 kernel size.
        self.pool = nn.MaxPool2d(2, 2)            # Max pooling layer with a 2x2 kernel size and a stride of 2.
        self.conv2 = nn.Conv2d(6, 16, 5)          # Convolutional layer with 6 input channels, 16 output channels, and a 5x5 kernel size.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)     # Fully connected layer with 1655 input nodes and 120 output nodes.
        self.fc2 = nn.Linear(120, 84)             # Fully connected layer with 120 input nodes and 84 output nodes.
        self.fc3 = nn.Linear(84, num_classes)     # Fully connected layer with 84 input nodes and 10 output nodes. In classification problems, this layer usually has the same number of nodes as the number of classes.

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # How the data flows through the network from input to output.
        x = self.pool(F.relu(self.conv1(x)))                # Process the input x through the first convolutional layer, applying the ReLU activation function, followed by max pooling.
        x = self.pool(F.relu(self.conv2(x)))                # Same for the second convolutional layer.
        x = x.view(-1, 16 * 5 * 5)                          # Flatten the output from the second pooling layer to prepare it for the fully connected layer. -1 means the size of that dimension is inferred to keep the total number of elements constant.
        x = F.relu(self.fc1(x))                             # Apply ReLU activation to the output of the first fully connected layer.
        x = F.relu(self.fc2(x))                             # Same for the second fully connected layer.
        return self.fc3(x)                                  # No activation function here, common in classification problems where the appropriate loss function (e.g., nn.CrossEntropyLoss) includes softmax internally.
    
    def get_params(self):
        model_params = [val.cpu().numpy() for _, val in self.state_dict().items()]
        return np.concatenate(model_params, axis=None).ravel()

    def set_params(self, updated_model):
        parameters = []
        init = 0
        for _, tensor_parameter in self.state_dict().items():
            end = init + tensor_parameter.numel()  # number of elements in tensor
            recovered_tensor = torch.tensor(updated_model[init:end], dtype=tensor_parameter.dtype)
            recovered_tensor = recovered_tensor.view(tensor_parameter.shape)
            parameters.append(recovered_tensor)
            init = end

        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)
        return
    
    def train_model(self, trainloader, epochs, DEVICE, show_progress):
        """Train the model on the training set."""
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.train()
        for _ in range(epochs):
            if show_progress is True:
                trainloader_aux = tqdm(trainloader)
            else:
                trainloader_aux = trainloader
            for images, labels in trainloader_aux:
                optimizer.zero_grad()
                if DEVICE is not None:
                    loss_fn(self(images.to(DEVICE)), labels.to(DEVICE)).backward()
                else:
                    loss_fn(self(images), labels).backward()
                optimizer.step()

    def test_model(self, testloader, DEVICE):
        """Validate the model on the test set."""
        loss_fn = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        self.eval()
        with torch.no_grad():
            for images, labels in testloader:
                if DEVICE is not None:
                    outputs = self(images.to(DEVICE))
                    labels = labels.to(DEVICE)
                else:
                    outputs = self(images)
                loss += loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        average_loss = loss / len(testloader.dataset)
        return accuracy, average_loss
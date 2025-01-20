import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm

import numpy as np

# Code modified from the Flower project on GitHub
# Repository link: https://github.com/adap/flower
# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader (MNIST)
# #############################################################################
class MNIST_Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: 
    A 60 Minute Blitz')"""
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
        """Train the network on the training set."""
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.train()

        for _ in range(epochs):
            if show_progress is True:
                trainloader_aux = tqdm(trainloader)
            else:
                trainloader_aux = trainloader
            for images, labels in trainloader_aux:
                optimizer.zero_grad()
                if DEVICE is not None:
                    loss_fn(self(images.to(DEVICE)), labels.to(DEVICE)).backward()         # Calcula la pérdida entre las predicciones del modelo y las etiquetas reales, y luego propaga hacia atrás los gradientes a través de la red neuronal.
                else:
                    loss_fn(self(images), labels).backward()         # Calcula la pérdida entre las predicciones del modelo y las etiquetas reales, y luego propaga hacia atrás los gradientes a través de la red neuronal.
                optimizer.step()
        
    def test_model(self, testloader, DEVICE):
        """Validate the network on the entire test set."""
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
        return {
            "accuracy": accuracy,
            "loss": average_loss
        }
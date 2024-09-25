import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
#from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader (CIFAR10)
# #############################################################################
class CIFAR10_Net(nn.Module):   # Define un modelo de red neuronal, nn.Module es la clase base para todos los modelos en PyTorch
    """Model (simple CNN adapted from 'PyTorch: 
    A 60 Minute Blitz')"""

    def __init__(self, num_classes: int) -> None:   # 62006 parameters with 10 classes
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)           # Capa de convolución con 3 canales de entrada, 6 canales de salida y un tamaño de kernel de 5x5.
        self.pool = nn.MaxPool2d(2, 2)            # Capa de pooling máxima con un tamaño de kernel de 2x2 y un stride de 2.
        self.conv2 = nn.Conv2d(6, 16, 5)          # Capa de convolución con 6 canales de entrada, 16 canales de salida y un tamaño de kernel de 5x5.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)     # Capa completamente conectada (fully connected) con 1655 nodos de entrada y 120 nodos de salida.
        self.fc2 = nn.Linear(120, 84)             # Capa completamente conectada con 120 nodos de entrada y 84 nodos de salida.
        self.fc3 = nn.Linear(84, num_classes)              # Capa completamente conectada con 84 nodos de entrada y 10 nodos de salida. En el contexto de redes neuronales para clasificación, esta última capa suele tener un número de nodos igual al número de clases en el problema.

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # Cómo los datos se propagan a través de la red desde la entrada hasta la salida.
        x = self.pool(F.relu(self.conv1(x)))                # Procesar la entrada x a través de la primera capa convolucional, aplicando la función de activación ReLU y luego realizando un pooling máximo.
        x = self.pool(F.relu(self.conv2(x)))                # Idem, siguiente capa convolucional
        x = x.view(-1, 16 * 5 * 5)                          # Aplanamiento de la salida de la segunda capa de pooling para prepararla para la capa completamente conectada. -1 (tamaño de ese eje debe inferirse para mantener el mismo número total de elementos)
        x = F.relu(self.fc1(x))                             # Aplica la función de activación ReLU a la salida de la primera capa completamente conectada
        x = F.relu(self.fc2(x))                             # Idem
        return self.fc3(x)                                  # No hay función de activación, común en problemas de clasificación donde se utiliza la función de pérdida adecuada (como nn.CrossEntropyLoss), que incluye la operación softmax internamente.


def train_CIFAR10(net, trainloader, epochs, DEVICE):
    """Train the model on the training set."""
    loss_fn = torch.nn.CrossEntropyLoss()                                   # Función de pérdida
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # Este optimizador ajustará los pesos del modelo durante el entrenamiento para minimizar la pérdida.
    net.train()
    for _ in range(epochs):                                                   # En cada epoca se recorre todo el conjunto de datos
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()                                                 # Inicializa los gradientes de todos los parámetros del modelo
            if DEVICE is not None:
                loss_fn(net(images.to(DEVICE)), labels.to(DEVICE)).backward()         # Calcula la pérdida entre las predicciones del modelo y las etiquetas reales, y luego propaga hacia atrás los gradientes a través de la red neuronal.
            else:
                loss_fn(net(images), labels).backward()         # Calcula la pérdida entre las predicciones del modelo y las etiquetas reales, y luego propaga hacia atrás los gradientes a través de la red neuronal.
            optimizer.step()                                                      # Actualiza los parámetros del modelo utilizando el optimizador, basándose en los gradientes calculados durante la fase de retropropagación

def test_CIFAR10(net, testloader, DEVICE):
    """Validate the model on the test set."""
    loss_fn = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():                                                       # desactiva el cálculo y seguimiento automático de gradientes
        for images, labels in testloader:
            if DEVICE is not None:
                outputs = net(images.to(DEVICE))
                labels = labels.to(DEVICE)
            else:
                outputs = net(images)
            loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()                       # Predicciones correctas en el lote. torch.max(·)[1] para obtener las clases predichas por el modelo.
    accuracy = correct / len(testloader.dataset)
    average_loss = loss / len(testloader.dataset)
    return accuracy, average_loss


def load_CIFAR10():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])        # Define una secuencia de transformaciones que se aplicarán a las imágenes
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return trainset, testset
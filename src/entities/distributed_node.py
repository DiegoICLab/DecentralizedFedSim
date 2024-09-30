import torch
import socket
import threading
import pickle
import numpy as np
from datetime import datetime
from queue import Queue

from collections import OrderedDict
from machine_learning.dataset.MNIST import MNIST_Net, train_MNIST, test_MNIST
from machine_learning.dataset.CIFAR10 import CIFAR10_Net, train_CIFAR10, test_CIFAR10
from utils.utils_logs import *

class DistributedNode:
    def __init__(self, node_id, ip, port, neighbors, dataset, trainloader, testloader, rounds):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.neighbors = neighbors

        self.dataset = dataset
        if self.dataset == "MNIST":
            self.model = MNIST_Net(num_classes=10)
        elif self.dataset == "CIFAR-10":
            self.model = CIFAR10_Net(num_classes=10)
        self.trainloader = trainloader
        self.testloader = testloader
        self.rounds = rounds
        
        self.model_queue = Queue()  # Queue to store received models
        self.listener_thread = None

        self.statistics = {
            "accuracy": [],
            "loss": [],
            "local_model": []
        }

    ##################################
    # Model functionalities
    ##################################   
    def get_parameters(self):
        model_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return np.concatenate(model_params, axis=None).ravel()

    def set_parameters(self, updated_model):
        parameters = []
        init = 0
        for _, tensor_parameter in self.model.state_dict().items():
            end = init + tensor_parameter.numel()  # number of elements in tensor
            recovered_tensor = torch.tensor(updated_model[init:end], dtype=tensor_parameter.dtype)
            recovered_tensor = recovered_tensor.view(tensor_parameter.shape)
            parameters.append(recovered_tensor)
            init = end

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        return

    # Function to train the local model for one round
    def train_local_model(self, epochs=1):
        if self.dataset == "MNIST":
            train_MNIST(self.model, self.trainloader, epochs, None)
        elif self.dataset == "CIFAR-10":
            train_CIFAR10(self.model, self.trainloader, epochs, None)
        return 
    
    # Function to evaluate the local model
    def evaluate_local_model(self):
        if self.dataset == "MNIST":
            return test_MNIST(self.model, self.testloader, None)
        elif self.dataset == "CIFAR-10":
            return test_CIFAR10(self.model, self.testloader, None)

    ##################################
    # Communication functionalities
    ##################################

    # Function to send model parameters to a neighbor
    def send_model(self, ip, port, model_state, round_number):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            # Serialize the model state dictionary
            model_state_bytes = pickle.dumps(model_state)
            timestamp = datetime.utcnow().isoformat()
            model_update = {
                'timestamp': timestamp,
                'node_id': self.node_id,
                'local_model': model_state_bytes,
                'round_number': round_number
            }
            sock.sendall(pickle.dumps(model_update))
            sock.close()
        except Exception as e:
            print(f"Error sending model to {ip}:{port} - {e}")

    # Function to receive model parameters from other nodes
    def receive_models(self):
        # TODO Create function to stop the threat
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.ip, self.port))
        sock.listen(20)

        while True:
            conn, addr = sock.accept()
            # conn.settimeout(5)
            data = b''
            while True:
                try:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                except socket.timeout:
                    log_info_node(self.node_id, "Timeout error")
                    break
            model_update = pickle.loads(data)
            model_update['local_model'] = pickle.loads(model_update['local_model'])
            self.model_queue.put(model_update) 
            conn.close()

    # Start a thread for receiving models
    def start_listener(self):
        self.listener_thread = threading.Thread(target=self.receive_models)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    # Function to get all updates from the queue
    def get_all_updates_from_queue(self):
        received_updates = []
        while not self.model_queue.empty():
            received_updates.append(self.model_queue.get())
        return received_updates
    
    # Function to get the number of updates in the queue
    def get_num_updates_queue(self):
        return self.model_queue.qsize()
    
    ##################################
    # Statistics
    ##################################

    def save_statistics(self):
        accuracy, loss = self.evaluate_local_model()
        self.statistics["accuracy"].append(accuracy)
        self.statistics["loss"].append(loss)
        self.statistics["local_model"].append(self.get_parameters())
        return
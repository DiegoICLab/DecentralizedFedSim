import torch
import socket
import threading
import pickle
import numpy as np
from datetime import datetime
from queue import Queue

from utils.utils_logs import *

class DistributedNode:
    def __init__(self, node_id, ip, port, neighbors, dataset, model, trainloader, testloader, rounds, conf_nodes):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.neighbors = neighbors

        self.dataset = dataset
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.rounds = rounds
        self.conf_nodes = conf_nodes
        self.conf_nodes['node_id'] = self.node_id
        
        self.model_queue = Queue()  # Queue to store received models
        self.listener_thread = None

        self.statistics = {}
        self.local_model_history = []

    ##################################
    # Model functionalities
    ##################################   
    def get_parameters(self):
        return self.model.get_params()

    def set_parameters(self, updated_model):
        self.model.set_params(updated_model)
        return

    # Function to train the local model for one round
    def train_local_model(self):
        self.model.train_model(self.trainloader, self.conf_nodes)
        return 
    
    # Function to evaluate the local model
    def evaluate_local_model(self):
        return self.model.test_model(self.testloader, self.conf_nodes)

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
        results = self.evaluate_local_model()
        
        for metric in results.keys():
            if metric not in self.statistics:
                self.statistics[metric] = []
                
        for key, value in results.items():
            self.statistics[key].append(value)
            
        self.local_model_history.append(self.get_parameters())
        return
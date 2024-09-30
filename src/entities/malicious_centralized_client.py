import time

from entities.centralized_client import CentralizeClient
from utils.utils_logs import *
from utils.utils_measures import *
from machine_learning.attacks.utils import(
    client_label_flipping_attack,
    client_sign_flipping_attack,
    client_noise_attack,
    client_IPM_attack,
    client_ALIE_attack
)

class MaliciousCentralizeClient(CentralizeClient):
    def __init__(self, node_id, ip, port, neighbors, server_id, dataset, trainloader, testloader, rounds, barrier_sim, byz_attack, attack_config ):
        super().__init__(
            node_id=node_id,
            ip=ip,
            port=port,
            neighbors=neighbors,
            server_id=server_id,
            dataset=dataset,
            trainloader=trainloader,
            testloader=testloader,
            rounds=rounds,
            barrier_sim=barrier_sim
        )
        self.byz_attack = byz_attack
        self.attack_config = attack_config

        if self.byz_attack == "Label-Flipping":
            # TODO Check 
            log_info_node(self.node_id, f"Computing {self.byz_attack} attack. Modifying 100% of the training labels.")
            self.trainloader = client_label_flipping_attack(self.trainloader, percentage_flip=1)
            
    # Main function to start the  centralized training process
    def run(self):
        # Start listener threads for each node
        self.start_listener()
        for round_num in range(self.rounds):
            # Training phase
            log_info_node(self.node_id, f"Round {round_num}. Starting training process...")
            self.train_local_model(epochs=1)

            # Wait for all client to finish the training phase
            log_info_node(self.node_id, f"Training finished. Waiting for other clients to complete training...")
            self.barrier_sim.wait()  # Simulated synchronize with other nodes

            # Compute malicious attack
            if self.byz_attack != "Label-Flipping":
                time.sleep(5)
                log_info_node(self.node_id, f"Computing Byzantine attack ({self.byz_attack})")
            received_updates = self.get_all_updates_from_queue()
            self.perform_model_poisoning(received_updates)
            
            # Send the model to neighbors
            log_info_node(self.node_id, f"Sending poisoned model to neighbors...")
            for neighbor in self.neighbors:
                self.send_model(neighbor['ip'], neighbor['port'], self.get_parameters(), round_num)

            # Receiving the global model from server
            received_server_model = False
            while True:
                time.sleep(5)
                if self.get_num_updates_queue() != 0:
                    received_updates = self.get_all_updates_from_queue()
                    for update in received_updates:
                        if str(update['node_id']) == self.server_id and update['local_model'] is not None:
                            received_server_model = True
                            server_model = update['local_model']
                            self.set_parameters(server_model)
                            log_info_node(self.node_id, f"Received global model from server.")
                            break   
                if received_server_model:
                    break
            
            # Save the model stats (optional)
            self.save_statistics()

        log_info_node(self.node_id, f"Training complete!")
    
    ############################################################## 
    # Performing model attacks
    ##############################################################
    def perform_model_poisoning(self, received_updates):

        received_models = [update['local_model'] for update in received_updates]
        if self.byz_attack == "Sign-Flipping":
            flipped_model = client_sign_flipping_attack(self.get_parameters())
            self.set_parameters(flipped_model)

        elif self.byz_attack == "Noise":
            noisy_model = client_noise_attack(self.get_parameters())
            self.set_parameters(noisy_model)
        
        elif self.byz_attack == "IPM":
            epsilon = self.attack_config["epsilon"]
            computed_model = client_IPM_attack(received_models, epsilon)
            self.set_parameters(computed_model)

        elif self.byz_attack == "ALIE":
            zmax = self.attack_config["zmax"]
            computed_model = client_ALIE_attack(received_models, zmax)
            self.set_parameters(computed_model)
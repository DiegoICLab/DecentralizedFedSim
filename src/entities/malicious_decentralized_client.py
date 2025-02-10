import time
import numpy as np

from entities.decentralized_client import DecentralizeClient
from utils.utils_logs import *
from utils.utils_measures import *
from machine_learning.attacks.utils import(
    client_label_flipping_attack,
    client_sign_flipping_attack,
    client_noise_attack,
    client_IPM_attack,
    client_ALIE_attack
)

class MaliciousDecentralizeClient(DecentralizeClient):
    def __init__(self, node_id, ip, port, neighbors, dataset, model, trainloader, testloader, rounds, aggregation_alg,  
                 aggregation_config, conf_nodes, barrier_sim, gossip_share, byz_attack, attack_config ):
        super().__init__(
            node_id=node_id,
            ip=ip,
            port=port,
            neighbors=neighbors,
            dataset=dataset,
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            rounds=rounds,
            aggregation_alg=aggregation_alg,
            aggregation_config=aggregation_config,
            conf_nodes=conf_nodes,
            barrier_sim=barrier_sim,
            gossip_share=gossip_share
        )
        self.byz_attack = byz_attack
        self.attack_config = attack_config

        if self.byz_attack == "Label-Flipping":
            # TODO Check 
            log_info_node(self.node_id, f"Computing {self.byz_attack} attack. Modifying 100% of the training labels.")
            self.trainloader = client_label_flipping_attack(self.trainloader, percentage_flip=1)

    # Main function to start the decentralized training process
    def run(self):
        # Start listener threads for each node
        self.start_listener()
        for round_num in range(self.rounds):
            log_info_node(self.node_id, f"Round {round_num}. Starting training process...")
            # Train the model locally
            self.train_local_model()

            # Wait for all nodes to finish the training round
            log_info_node(self.node_id, f"Training finished. Waiting for other nodes to complete training...")
            self.barrier_sim.wait()  # Synchronize with other nodes

            # Simulate sharing time interval
            # With this simulated time sharing, model attack will be computed by using only the received models
            if self.byz_attack != "Label-Flipping":
                time.sleep(5)
                log_info_node(self.node_id, f"Computing Byzantine attack ({self.byz_attack})")
            received_updates = self.get_all_updates_from_queue()
            self.perform_model_poisoning(received_updates)

            log_info_node(self.node_id, f"Sending poisoned model to neighbors...")
            # Send the model to neighbors
            if self.gossip_share == False:
                for neighbor in self.neighbors:
                    self.send_model(neighbor['ip'], neighbor['port'], self.get_parameters(), round_num)
            else:
                # Gossip sharing
                random_neighbor = np.random.choice(self.neighbors)
                self.send_model(random_neighbor['ip'], random_neighbor['port'], self.get_parameters(), round_num)
            
            # Simulate sharing time interval
            # With this simulated time sharing, model sharing must be reciprocal
            wait = 2
            while True:
                if wait == 0:
                    break
                time.sleep(5)
                if self.get_num_updates_queue() == len(self.neighbors):
                    break
                wait -= 1
            
            # Aggregate the received models into the local model
            num_received_models = len(received_updates)
            log_info_node(self.node_id, f"Aggregating models ({self.aggregation_alg})... Received {num_received_models} models.")
            aggregated_model = self.aggregation_models(received_updates, round_num)
            if aggregated_model is not None:
                self.set_parameters(aggregated_model)

            # Save the model stats (optional)
            self.save_statistics()
            if self.node_id in self.conf_nodes["show_results_round"]:
                log_info(f"Showing results of this round of node {self.node_id} as an example:")
                for key, value in self.statistics.items():
                    acc = self.statistics[key][-1]
                    log_info(f"{key}: {acc}")
                    
                model = self.local_model_history[-1]
                log_info(f"local model: {model}")

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
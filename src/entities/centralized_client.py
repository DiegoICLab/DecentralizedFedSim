import time
import numpy as np

from entities.distributed_node import DistributedNode
from machine_learning.aggregation.byzantine_robust_algorithms import (
    Mean_aggregation, 
    Median_aggregation, 
    MultiKrum_aggregation,
    TrimmedMean_aggregation,
    Clustering_aggregation,
    WFAgg_D_aggregation,
    WFAgg_C_aggregation,
    WFAgg_T_aggregation,
    WFAgg_E_aggregation,
    Alt_WFAgg_aggregation,
    WFAgg_aggregation,
    compute_euclidean_distance,
    compute_cosine_similarity
)
from utils.utils_logs import *
from utils.utils_measures import *

class CentralizeClient(DistributedNode):
    def __init__(self, node_id, ip, port, neighbors, server_id, dataset, trainset, testset, rounds, barrier_sim ):
        super().__init__(
            node_id=node_id,
            ip=ip,
            port=port,
            neighbors=neighbors,
            dataset=dataset,
            trainset=trainset,
            testset=testset,
            rounds=rounds
        )
        self.server_id=server_id
        self.barrier_sim = barrier_sim

    # Main function to start the decentralized training process
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

            # Send the model to neighbors
            log_info_node(self.node_id, f"Sending model to neighbors...")
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
        # num_bytes = get_length_bytes(self.get_parameters())
        # log_info_node(self.node_id, f"Stored statistics information: {num_bytes} bytes")
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

class ParameterServer(DistributedNode):
    def __init__(self, node_id, ip, port, neighbors, dataset, testset, rounds, aggregation_alg,  
                 aggregation_config ):
        super().__init__(
            node_id=node_id,
            ip=ip,
            port=port,
            neighbors=neighbors,
            dataset=dataset,
            trainset=None,
            testset=testset,
            rounds=rounds
        )
        self.aggregation_alg = aggregation_alg
        self.aggregation_config = aggregation_config

        self.previous_neighbor_models = {}
        self.temporal_metrics = {
            "distance": {},
            "cosine": {}
        }

    # Main function to start the decentralized training process
    def run(self):
        # Start listener threads for each node
        self.start_listener()
        for round_num in range(self.rounds):
            # Training phase
            log_info_node(self.node_id, f"Round {round_num}. Waiting for model updates from clients...")
            
            # Simulate sharing time interval
            while True:
                time.sleep(5)
                if self.get_num_updates_queue() == len(self.neighbors):
                    break
            
            # Get all received models in this round
            received_updates = self.get_all_updates_from_queue()
            num_received_models = len(received_updates)
            log_info_node(self.node_id, f"Aggregating models ({self.aggregation_alg})... Received {num_received_models} models.")
            # Aggregate the received models into the local model
            aggregated_model = self.aggregation_models(received_updates, round_num)
            if aggregated_model is not None:
                self.set_parameters(aggregated_model)

            # Send the model to neighbors
            log_info_node(self.node_id, f"Sending model to neighbors...")
            for neighbor in self.neighbors:
                self.send_model(neighbor['ip'], neighbor['port'], self.get_parameters(), round_num)

            # Save the model stats (optional)
            self.save_statistics()
            acc = self.statistics["accuracy"][-1]
            model = self.statistics["local_model"][-1]
            log_info(f"Accuracy: {acc}")
            log_info(f"Model: {model}")

        log_info_node(self.node_id, f"Training complete!")
        # num_bytes = get_length_bytes(self.get_parameters())
        # log_info_node(self.node_id, f"Stored statistics information: {num_bytes} bytes")

    # Function to average the local model with received models
    def aggregation_models(self, received_updates, round):
        if not received_updates:
            return self.get_parameters()  # If no models received, return the local model

        neighbor_ids = [update['node_id'] for update in received_updates]
        received_models = [update['local_model'] for update in received_updates]

        # Statistics-based SOTA algorithms
        if self.aggregation_alg == "Mean":
            aggregated_model = Mean_aggregation(received_models)            

        elif self.aggregation_alg == "Median":
            aggregated_model = Median_aggregation(received_models)

        elif self.aggregation_alg == "Trimmed-Mean":
            aggregated_model = TrimmedMean_aggregation(received_models, float(self.aggregation_config["proportiontocut"]))
        
        # Distance-based SOTA algorithms
        elif self.aggregation_alg == "Krum":
            num_malicious = int(self.aggregation_config["num_malicious"])
            aggregated_model, best_indices = MultiKrum_aggregation(None, received_models, num_malicious, 0)
            selected_ids = [ neighbor_ids[i] for i in best_indices ]
            log_info_node(self.node_id, f"Selected neighbor models for Krum: {selected_ids}.")
        
        elif self.aggregation_alg == "Multi-Krum":
            num_malicious = int(self.aggregation_config["num_malicious"])
            to_keep = int(self.aggregation_config["to_keep"])
            aggregated_model, best_indices = MultiKrum_aggregation(None, received_models, num_malicious, to_keep)
            selected_ids = [ neighbor_ids[i] for i in best_indices ]
            log_info_node(self.node_id, f"Selected neighbor models for Multi-Krum: {selected_ids}.")

        elif self.aggregation_alg == "Clustering":
            aggregated_model, best_indices = Clustering_aggregation(None, received_models)
            selected_ids = [ neighbor_ids[i] for i in best_indices ]
            log_info_node(self.node_id, f"Selected neighbor models for Clustering: {selected_ids}.")

        # Proposed algorithms
        elif self.aggregation_alg == "WFAgg-D":
            num_malicious = int(self.aggregation_config["num_malicious"])
            aggregated_model, best_indices = WFAgg_D_aggregation(None, received_models, num_malicious)
            selected_ids = [ neighbor_ids[i] for i in best_indices ]
            log_info_node(self.node_id, f"Selected neighbor models for WFAgg-D: {selected_ids}.")

        elif self.aggregation_alg == "WFAgg-C":
            num_malicious = int(self.aggregation_config["num_malicious"])
            aggregated_model, best_indices = WFAgg_C_aggregation(None, received_models, num_malicious)
            selected_ids = [ neighbor_ids[i] for i in best_indices ]
            log_info_node(self.node_id, f"Selected neighbor models for WFAgg-C: {selected_ids}.")
        
        elif self.aggregation_alg == "WFAgg-T":
            transitory_rounds = int(self.aggregation_config["transitory_rounds"])
            previous_models = [ self.previous_neighbor_models[id] if id in list(self.previous_neighbor_models.keys()) else None  
                                for id in neighbor_ids ]
            self.update_temporal_statistics(neighbor_ids, received_models, previous_models, round, transitory_rounds)

            aggregated_model, best_indices = WFAgg_T_aggregation(
                None, 
                received_models,
                [self.temporal_metrics["distance"][id] for id in neighbor_ids],
                [self.temporal_metrics["cosine"][id] for id in neighbor_ids],
                True
            )
            selected_ids = [ neighbor_ids[i] for i in best_indices ]
            log_info_node(self.node_id, f"Selected neighbor models for WFAgg-T: {selected_ids}.")
        
        elif self.aggregation_alg == "WFAgg-E":
            smooth_factor = float(self.aggregation_config["smooth_factor"])
            aggregated_model = WFAgg_E_aggregation(self.get_parameters(), received_models, smooth_factor, np.ones(len(received_models)))

        elif self.aggregation_alg == "Alt-WFAgg":
            transitory_rounds = int(self.aggregation_config["transitory_rounds"])
            num_malicious = int(self.aggregation_config["num_malicious"])
            smooth_factor = float(self.aggregation_config["smooth_factor"])

            previous_models = [ self.previous_neighbor_models[id] if id in list(self.previous_neighbor_models.keys()) else None  
                                for id in neighbor_ids ]
            self.update_temporal_statistics(neighbor_ids, received_models, previous_models, round, transitory_rounds)

            aggregated_model, weights = Alt_WFAgg_aggregation(self.get_parameters(), received_models, num_malicious, smooth_factor,
                                                              [self.temporal_metrics["distance"][id] for id in neighbor_ids],
                                                              [self.temporal_metrics["cosine"][id] for id in neighbor_ids])
            log_info_node(self.node_id, f"Neighbor models {neighbor_ids} for aggregation have weights: {weights}.")

        elif self.aggregation_alg == "WFAgg":
            transitory_rounds = int(self.aggregation_config["transitory_rounds"])
            num_malicious = int(self.aggregation_config["num_malicious"])
            smooth_factor = float(self.aggregation_config["smooth_factor"])

            previous_models = [ self.previous_neighbor_models[id] if id in list(self.previous_neighbor_models.keys()) else None  
                                for id in neighbor_ids ]
            self.update_temporal_statistics(neighbor_ids, received_models, previous_models, round, transitory_rounds)

            aggregated_model, weights = WFAgg_aggregation(self.get_parameters(), received_models, num_malicious, smooth_factor,
                                                          [self.temporal_metrics["distance"][id] for id in neighbor_ids],
                                                          [self.temporal_metrics["cosine"][id] for id in neighbor_ids])
            log_info_node(self.node_id, f"Neighbor models {neighbor_ids} for aggregation have weights: {weights}.")

        else:
            log_error(f"Este algoritmo de agregacion robusta no está implementado. Se mantiene el mismo modelo de la anterior ronda.")
            aggregated_model = None

        return aggregated_model

    def update_temporal_statistics(self, id_neighbors, current_models, previous_models, round, transitory_rounds):

        for i, id in enumerate(id_neighbors):
            self.previous_neighbor_models[id] = current_models[i]

            if id not in list(self.temporal_metrics["distance"].keys()):
                self.temporal_metrics["distance"][id] = []
            
            if id not in list(self.temporal_metrics["cosine"].keys()):
                self.temporal_metrics["cosine"][id] = []

            if previous_models[i] is not None and round >= transitory_rounds:
                self.temporal_metrics["distance"][id].append( compute_euclidean_distance(current_models[i], previous_models[i]) )
                self.temporal_metrics["cosine"][id].append( compute_cosine_similarity(current_models[i], previous_models[i]) )
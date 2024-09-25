import threading

from entities.decentralized_client import DecentralizeClient
from entities.malicious_decentralized_client import MaliciousDecentralizeClient
from utils.utils_logs import *
from machine_learning.metrics.utils import R_squared_models

from typing import List, Dict, Any

def decentralized_simulation(
    sim_config: Dict[str, Any], 
    nodes_config: Dict[str, Any], 
    trainloaders, 
    valloaders
    ) -> Dict[str, Any]:
    """
    Simulates a decentralized federated learning process across multiple nodes.

    Args:
        sim_config: General simulation configuration (e.g., global hyperparameters).
        nodes_config: Specific configuration for each node.
        trainloaders: List of data loaders for training on each node.
        valloaders: List of data loaders for validation on each node.

    Returns:
        simulation_results: Results of the simulation, which may include performance metrics, statistics, etc.
    """

    nodes = {}
    barrier_sim = threading.Barrier(len(nodes_config))      # Synchronization barrier for nodes

    for index, (key, value) in enumerate(nodes_config.items()):

        if value['id'] in sim_config["malicious_nodes"]:
            # Create instances of MaliciousDecentralizeNode
            nodes[key] = MaliciousDecentralizeClient(
                node_id=value['id'], 
                ip=value['ip'], 
                port=value['port'], 
                neighbors= [nodes_config[str(i)] for i in value['neighbors']], 
                dataset=sim_config["dataset"],
                trainset=trainloaders[index],
                testset=valloaders[index],
                rounds=sim_config["rounds"],
                aggregation_alg=sim_config["algorithm"],
                aggregation_config=sim_config.get("algorithm_config", None),
                barrier_sim=barrier_sim,
                byz_attack=sim_config["byz_attack"],
                attack_config=sim_config.get("attack_config", None)
            )
        else:
            # Create instances of DecentralizeNode
            nodes[key] = DecentralizeClient(
                node_id=value['id'], 
                ip=value['ip'], 
                port=value['port'], 
                neighbors= [nodes_config[str(i)] for i in value['neighbors']], 
                dataset=sim_config["dataset"],
                trainset=trainloaders[index],
                testset=valloaders[index],
                rounds=sim_config["rounds"],
                aggregation_alg=sim_config["algorithm"],
                aggregation_config=sim_config.get("algorithm_config", None),
                barrier_sim=barrier_sim
            )
        
    # Run the training process in separate threads for each node
    threads = []
    for key, decentralized_node in nodes.items():
        t = threading.Thread(target=decentralized_node.run)
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    simulation_results = {
        "accuracy": {},
        "loss": {},
        "R-Squared": []
    }
    
    for key, node in nodes.items():
        simulation_results["accuracy"][key] = node.statistics["accuracy"]
        simulation_results["loss"][key] = node.statistics["loss"]
    
    for round in range(int(sim_config["rounds"])):
        models_round = []
        for _, node in nodes.items():
            models_round.append(node.statistics["local_model"][round])
        simulation_results["R-Squared"].append(R_squared_models(models_round))
    
    return simulation_results


def centralized_simulation(
    sim_config: Dict[str, Any], 
    nodes_config: Dict[str, Any], 
    trainloaders, 
    valloaders,
    testloader
    ) -> Dict[str, Any]:
    """
    Simulates a centralized federated learning process across multiple nodes and a server.

    Args:
        sim_config: General simulation configuration (e.g., global hyperparameters).
        nodes_config: Specific configuration for each node.
        trainloaders: List of data loaders for training on each node.
        valloaders: List of data loaders for validation on each node.
        testloader: Data loader for validation on server.
    Returns:
        simulation_results: Results of the simulation, which may include performance metrics, statistics, etc.
    """

    nodes = {}
    barrier_sim = threading.Barrier(len(nodes_config)-1)      # Synchronization barrier for nodes

    for index, (key, value) in enumerate(nodes_config.items()):

        if value['id'] == sim_config["server_id"]:
            print("")
        
        
        elif value['id'] in sim_config["malicious_nodes"]:
            # Create instances of MaliciousDecentralizeNode
            nodes[key] = MaliciousDecentralizeClient(
                node_id=value['id'], 
                ip=value['ip'], 
                port=value['port'], 
                neighbors= [nodes_config[str(i)] for i in value['neighbors']], 
                dataset=sim_config["dataset"],
                trainset=trainloaders[index],
                testset=valloaders[index],
                rounds=sim_config["rounds"],
                aggregation_alg=sim_config["algorithm"],
                aggregation_config=sim_config.get("algorithm_config", None),
                barrier_sim=barrier_sim,
                byz_attack=sim_config["byz_attack"],
                attack_config=sim_config.get("attack_config", None)
            )
        else:
            # Create instances of DecentralizeNode
            nodes[key] = DecentralizeClient(
                node_id=value['id'], 
                ip=value['ip'], 
                port=value['port'], 
                neighbors= [nodes_config[str(i)] for i in value['neighbors']], 
                dataset=sim_config["dataset"],
                trainset=trainloaders[index],
                testset=valloaders[index],
                rounds=sim_config["rounds"],
                aggregation_alg=sim_config["algorithm"],
                aggregation_config=sim_config.get("algorithm_config", None),
                barrier_sim=barrier_sim
            )
        
    # Run the training process in separate threads for each node
    threads = []
    for key, decentralized_node in nodes.items():
        t = threading.Thread(target=decentralized_node.run)
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    simulation_results = {
        "accuracy": {},
        "loss": {},
        "R-Squared": []
    }
    
    for key, node in nodes.items():
        simulation_results["accuracy"][key] = node.statistics["accuracy"]
        simulation_results["loss"][key] = node.statistics["loss"]
    
    for round in range(int(sim_config["rounds"])):
        models_round = []
        for _, node in nodes.items():
            models_round.append(node.statistics["local_model"][round])
        simulation_results["R-Squared"].append(R_squared_models(models_round))
    
    return simulation_results
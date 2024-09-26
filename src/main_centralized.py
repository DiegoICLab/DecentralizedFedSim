import argparse
import json

from utils.utils_logs import *
from machine_learning.dataset.MNIST import load_MNIST
from machine_learning.dataset.CIFAR10 import load_CIFAR10
from machine_learning.dataset.utils import prepare_dataset
from simulator.simulation import centralized_simulation

def main(args, nodes_config):

    sim_config = {
        "rounds": args.rounds,
        "server_id": args.server,
        "algorithm": args.algorithm,
        "dataset": args.dataset,
        "malicious_nodes": args.malicious,
        "byz_attack": args.attack,
        "algorithm_config": {},
        "attack_config": {},
    }

    # TODO Move check type here
    if sim_config["byz_attack"] == "IPM":
        sim_config["attack_config"]["epsilon"] = args.paramsAtk[0]
    elif sim_config["byz_attack"] == "ALIE":
        sim_config["attack_config"]["zmax"] = args.paramsAtk[0]

    if sim_config["algorithm"] == "Trimmed-Mean":
        sim_config["algorithm_config"]["proportiontocut"] = args.paramsAgg[0]
    elif sim_config["algorithm"] == "Krum":
        sim_config["algorithm_config"]["num_malicious"] = args.paramsAgg[0]
    elif sim_config["algorithm"] == "Multi-Krum":
        sim_config["algorithm_config"]["num_malicious"] = args.paramsAgg[0]
        sim_config["algorithm_config"]["to_keep"] = args.paramsAgg[1]
    elif sim_config["algorithm"] == "WFAgg-D" or sim_config["algorithm"] == "WFAgg-C":
        sim_config["algorithm_config"]["num_malicious"] = args.paramsAgg[0]
    elif sim_config["algorithm"] == "WFAgg-T":
        sim_config["algorithm_config"]["transitory_rounds"] = args.paramsAgg[0]
    elif sim_config["algorithm"] == "WFAgg-E":
        sim_config["algorithm_config"]["smooth_factor"] = args.paramsAgg[0]
    elif sim_config["algorithm"] == "WFAgg" or sim_config["algorithm"] == "Alt-WFAgg":
        sim_config["algorithm_config"]["num_malicious"] = args.paramsAgg[0]
        sim_config["algorithm_config"]["transitory_rounds"] = args.paramsAgg[1]
        sim_config["algorithm_config"]["smooth_factor"] = args.paramsAgg[2]

    if sim_config["dataset"] == "MNIST":
        trainset, testset = load_MNIST()
    elif sim_config["dataset"] == "CIFAR-10":
        trainset, testset = load_CIFAR10()

    trainloaders, valloaders, testloader = prepare_dataset(
        trainset, testset, num_partitions=(len(nodes_config)-1), batch_size_client=32, batch_size_test=128, val_ratio=0.1
    )
    
    simulation_results = centralized_simulation(sim_config=sim_config, nodes_config=nodes_config, trainloaders=trainloaders, valloaders=valloaders, testloader=testloader)
    log_success(simulation_results)
    # file_name = config.output

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Program for the simulation of decentralized federated learning environments")
    parser.add_argument("-r", "--rounds", type=int, help="Number of rounds of communication", required=True)
    parser.add_argument("-s", "--server", type=str, help="ID of the server in the network topology", required=True)
    parser.add_argument("-t", "--topology", type=str, help="Path to the file containing the network topology", required=True)
    parser.add_argument("-m", "--malicious", type=int, nargs='+', help="Byzantine Node Array", required=False, default=[])
    parser.add_argument("-at", "--attack", type=str, help="Type of byzantine attack", required=False,
                        choices=["Label-Flipping", "Sign-Flipping", "Noise", "ALIE", "IPM"])
    parser.add_argument("-a", "--algorithm", type=str, help="Selected byzantine-robust algorithm", required=True,
                        choices=["Mean", "Krum", "Median", "Multi-Krum", "Trimmed-Mean", "Clustering", "WFAgg-D", "WFAgg-C", "WFAgg-T", "WFAgg-E", "Alt-WFAgg", "WFAgg"])
    parser.add_argument("-db", "--dataset", type=str, help="Selected dataset for this simulation", required=True,
                        choices=["MNIST", "CIFAR-10"])
    parser.add_argument("-p1", "--paramsAtk", type=float, nargs='+', help="Parameters for attack config", required=False, default=[])
    parser.add_argument("-p2", "--paramsAgg", type=float, nargs='+', help="Parameters for aggregation config", required=False, default=[])
    parser.add_argument("-o", "--output", type=str, help="Name of the output file", required=True)

    args = parser.parse_args()

    try:
        with open(args.topology, 'r') as file:
            nodes_config = json.load(file)
    except FileNotFoundError:
        log_error(f"No se ha encontrado el archivo con la topologia de la red")
        exit(1)

    if (len(args.malicious) == 0 and args.attack is not None) or (len(args.malicious) != 0 and args.attack is None):
        log_error(f"Revisar la configuración de los ataques y nodos maliciosos")
        exit(1)

    # for i in args.malicious:
    #     if i not in topology.keys():
    #         log_error(f"Algunos de los nodos maliciosos no se encuentran en la topologia")
    #         exit(1)
    
    main(args, nodes_config)
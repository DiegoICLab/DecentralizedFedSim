# DecentralizedFedSim: a DFL Environment Simulation Project

This project aims to simulate environments for **Decentralized Federated Learning** (DFL), a distributed learning technique where multiple nodes collaborate among them (without a central server) to train a Machine Learning model without sharing their local data. The simulation allows exploration of various network topologies, model aggregation strategies, and robustness against malicious nodes.

## Project Structure

- **/config**: Configuration files for the simulation (communication network).
- **/src**: Contains the simulation source code.
    -  **../entities**: Provides the main classes that define the role of the device in the learning network (server, client...)
    -  **../machine_learning**: Various tools for handling ML models, Byzantine attacks, aggregation algorithms...
    -  **../utils**: Multipurpose tools: logger...
    -  **../main_(de)centralized.py**: Provides a specific example of how to launch a (D)FL simulation.

## Requirements

- **Python 3.10.12**
- Libraries:
  - torch==2.0.1
  - torchvision==0.15.2
  - numpy==1.24.1
  - pandas==1.5.3
  - scipy==1.11.3
  - tqdm==4.65.0
  - seaborn==0.13.0
  - argparse==1.1
  - matplotlib==3.7.1
  - mpltex==0.7
  - colorama

## Installation

Clone the repository:

```bash
git clone https://github.com/DiegoICLab/DecentralizedFedSim.git
```

Navigate to the project directory:

```bash
cd DecentralizedFedSim
```

Install the necessary packages:

```bash
pip3 install -r requirements.txt
```

## Usage
To run the simulation, use the following command (it is one example):

```bash
python3 ./src/main_decentralized.py -r 10 -t './config/basic_decentralized_topology.json' -m 4 -at "IPM" -p1 0.5 -a "Trimmed-Mean" -p2 0.5 -db "MNIST" -o "decentralized_trimmed_mean_0.1_alie"

python3 ./src/main_centralized.py -r 10 -s 1 -t './config/basic_centralized_topology.json' -m 4 -at "IPM" -p1 0.5 -a "Trimmed-Mean" -p2 0.5 -db "MNIST" -o "centralized_trimmed_mean_0.1_alie"
```

Configuration parameters (more details in the source code):

- Communication rounds (`-r`): Number of rounds to train and share the ML model.
- Network topology (`-t`): Path to the file containing the network topology.
- Malicious nodes (`-m`): Byzantine Nodes Array.
- Byzantine attack (`-at`): Type of Byzantine attack
- Attack parameters (`-p1`): Parameters for attack config
- Aggregation algorithm (`-a`): Selected Byzantine-robust algorithm
- Algorithm parameters (`-p2`): Parameters for aggregation scheme config.
- Database (`-db`): Selected dataset for this simulation.
- Output file (`-o`): name of the output file.
- Server (`-s`): ID of the server in the network topology (only in centralized simulations)

## Future Work and Desired Contributions

We are actively seeking contributions in the following areas:

- **Scalability Improvements**: Enhancing the simulation to handle a larger number of nodes and more complex network topologies. It also aims to implement communication and coordination protocols that allow simulations in real environments with different devices.
- **New Aggregation Strategies**: Implementing and testing additional model aggregation methods that could improve the accuracy and robustness of the federated learning process.

If you have ideas for additional features, or if you'd like to collaborate on any of the topics listed above, we encourage you to contribute. Please follow the guidelines outlined in the **Contributions** section.

## Contributions

Contributions are welcome. To contribute, please follow these steps:

1. Fork the project.
2. Create a branch for your feature (git checkout -b feature/new-feature).
3. Commit your changes (git commit -m 'Added new feature').
4. Push to the branch (git push origin feature/new-feature).
5. Open a Pull Request.

## License

This project is licensed under the GNU GPLv3 License. See the LICENSE file for more details.

## Contact

For questions or suggestions, contact the author at dcajaraville@det.uvigo.es.
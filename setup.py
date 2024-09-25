from setuptools import setup, find_packages

setup(
    name='DecentralizedFedSim',
    version='2.0.0',
    author='dcajaraville',
    author_email='dcajaraville@det.uvigo.es',
    long_description='This project aims to simulate environments for **Decentralized Federated Learning** (DFL), a distributed learning technique where multiple nodes collaborate among them (without a central server) to train a Machine Learning model without sharing their local data. The simulation allows exploration of various network topologies, model aggregation strategies, and robustness against malicious nodes.',
    url='https://github.com/DiegoICLab/DecentralizedFedSim',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'torch==2.0.1',
        'torchvision==0.15.2',
        'numpy==1.24.1',
        'pandas==1.5.3',
        'scipy==1.11.3',
        'tqdm==4.65.0',
        'seaborn==0.13.0',
        'argparse==1.1',
        'matplotlib==3.7.1',
        'mpltex==0.7',
        'colorama'
    ],
    classifiers=[
        'License :: OSI Approved :: GNU GPLv3',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10.12',
    ],
)
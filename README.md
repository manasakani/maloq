# amorphous_gnns
Model to predict the electronic structure matrices of large amorphous materials.
The network architecture is directly adapted from EquiformerV2, which is located at the repository: []

Developers: [anonimized]

1. Set up the python environment with [use requirements.txt]
2. Download the datasets provided at []

To train an H2O molecule:

1. Enter /structures/molecules/a-HfO2/
2. To train the network: run train.py --f [path/to/datasets]
3. To test the network: run test.py --f [path/to/datasets] 

To train an HfO2 material:

1. Enter /structures/materials/H2O/
2. To train the network: run train.py --f [path/to/datasets]
3. To test the network: run test.py --f [path/to/datasets] 



# amorphous_gnns
Model to predict electronic structure matrices (Hamiltonians), with distributed compute functionality enabled
The network architecture is directly adapted from EquiformerV2, which is located at the repository: []

Note: This is not intended to be a fully usable repo just yet, but feel free to contact us if you're interested

Developers: [anonimized] 

1. Set up the python environment with [use requirements.txt]
2. Download the datasets provided at [provided after publication]

To train an H2O molecule:

1. Enter /structures/molecules/a-HfO2/
2. To train the network: run train.py --f [path/to/datasets]
3. To test the network: run test.py --f [path/to/datasets] 

To train an HfO2 material:

1. Enter /structures/materials/H2O/
2. To train the network: run train.py --f [path/to/datasets]
3. To test the network: run test.py --f [path/to/datasets] 

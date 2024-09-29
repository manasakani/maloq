# amorphous_gnns
Model to predict the electronic structure matrices of large amorphous materials.
The network architecture is directly adapted from EquiformerV2, which is located at the repository: []

Developers: Chen Hao Xia, Manasa Kaniselvan

1. Set up the python environment with [use requirements.txt]
2. Download the datasets provided at []

To train an H2O molecule:

1. Enter /structures/molecules/a-HfO2/train
2. To train the network: run main.py --f [path/to/datasets] --run train
3. To test the network: run main.py --f [path/to/datasets] --run test

To train an HfO2 material:

1. Enter /structures/materials/a-HfO2/train


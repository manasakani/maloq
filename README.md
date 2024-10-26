# amorphous_gnns
Model to predict the electronic structure matrices of large amorphous materials.
The network architecture is directly adapted from EquiformerV2, which is located at the repository: []

Note: this version is the 'minimal code' which will be used for the SC project

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

Note: this version of the code was set up to submitfor peer review, with some examples (in /structures) which can be run as tests. However it will still undergo extensive testing (particularily in a distributed environment) and further refactoring before the corresponding repository is made public.

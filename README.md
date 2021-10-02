# Handwritten-numbers-recognition
Introductory project to teach myself concepts of Neural Networks. Project utilizes ANNs to import pixel images of handwritten numbers from the MNIST dataset and interpret numerical values of the same.

The script named ANN contains the network class with its Initialization, Forward Propagate and Train (Back Propagate) functions. The script predominantly follows the tutorials given in "Make Your Own Neural Network"
by Tariq Rashid. However, there are some variables in the class that I added in to introduce Epoch iteration capability.
  - The function TrainNetwork iterates through the entire contents of the input MNIST file to back propagate error corrections.
  - There are a number of other functions that I wrote for file handling, generating custom image data (testing my own handwriting) and getting performance

The script HDF5 is a set of functions just to store the networks created before and after training cycles. The networks are stored in the HDF5 standard format (used in a lot of large scale scientific data storage). I prefer to use HDF5 to take advantage of its hierarchical data storage format.
All functions are for the sole purpose of file handling (creation, storage, branch handling, deletion, reading, etc.)

The main script contains functions to iterate through Epoch cycles. When called, EpochPlots delivers a plot representing Epoch vs performance for 10 different networks (Each with its own input, hidden and outer layer properties).

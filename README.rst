=========================================================
MLP Network for Digit Recognition using MNIST 784 dataset
=========================================================

Overview
========
This project implements a **Multi-Layer Perceptron (MLP)** from scratch in Python, using the **MNIST dataset** for classifying handwritten digits (0-9). The model is built with basic operations using `NumPy` and is modularized into separate components such as activation functions, forward and backward propagation, loss computation, and training. The model is evaluated based on accuracy on the MNIST dataset.

The project does not rely on machine learning frameworks like TensorFlow or PyTorch but implements the MLP with manual backpropagation.

Features
========
- MLP neural network with a single hidden layer.
- Modular design with separate files for activations, loss functions, and training logic.
- Training and evaluation functionality implemented from scratch.
- Basic matrix operations using `NumPy`.
- Utilizes `scikit-learn` for data preprocessing and train-test splitting.
- Achieves competitive accuracy on the MNIST dataset.

Installation
============
You can set up the project by installing the required dependencies and running the training script.

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/yourusername/digit-classifier.git
    cd digit-classifier

2. Install the dependencies via `pip`:

.. code-block:: bash

    pip install -r requirements.txt

Or, alternatively, using `setup.py`:

.. code-block:: bash

    pip install .

Usage
=====

1. Training the model:

After installing the dependencies, you can train the model by running the main script:

.. code-block:: bash

    python main.py

This will preprocess the MNIST data, initialize the MLP, and train it over multiple epochs.

2. Modifying the Model:

- You can adjust the **number of neurons**, **learning rate**, and **number of epochs** by editing the hyperparameters in the `main.py` script.
- The model architecture (layers, activations) can be modified in the `module/core/core.py` file and `main.py`.

Code Structure
==============

The project is divided into several modules:

- **module**: Contains the core network/model definition, including forward propagation, backpropagation, and weight updates.
- **activations**: Defines activation functions (`ReLU`, `softmax`) and their derivatives.
- **loss**: Contains the cross-entropy loss function for classification tasks.
- **trainer**: Handles training and evaluation logic for the MLP model.
- **utils**: Preprocessing functions like one-hot encoding and train-test splitting.

Testing
=======

Unit tests are available to verify the correctness of the MLP implementation, activations, and loss function. To run the tests:

.. code-block:: bash

    python -m unittest discover

License
=======
This project is licensed under the MIT License. See the LICENSE file for more details.

Author
======
Created by Sebastian Mandal.

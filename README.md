# Tensorized Deep Physics Informed Neural Networks :atom:

This repository contains an implementation of Tensorized Physics Informed Neural Networks (TPINNs) for solving physics-based problems. TPINNs combine the power of neural networks with the physical laws governing the system to improve accuracy and generalization. :rocket:

## :books: Introduction
Tensorized Physics Informed Neural Networks (TPINNs) are a class of neural networks that incorporate known physical laws into their architecture. By including the governing equations of a system as constraints, TPINNs can solve complex physics-based problems more accurately than traditional neural networks. This repository provides an implementation of TPINNs using TensorFlow. :infinity:

## :heavy_check_mark: Requirements
To run the code in this repository, you need the following dependencies:
- Python (>= 3.6)
- TensorFlow (>= 2.0)
- NumPy (>= 1.18)

## :floppy_disk: Installation
1. Clone the repository:
   ```
   git clone git@github.com:mvanzulli/TPINN.git
   ```
2. Navigate to the project directory:
   ```
   cd TPINN
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## :computer: Usage
To utilize TPINNs in your own projects, follow these steps:

1. Import the necessary modules:
   ```python
   import tensorflow as tf
   import numpy as np
   from tn_layer import TNLayer
   from tn_model import TNModel
   ```

2. Create an instance of the TNLayer class:
   ```python
   tn_layer = TNLayer(input_dim, bond_dim, activation, kernel_initializer, use_bias, bias_initializer)
   ```

   Replace the arguments with the desired values. `input_dim` is the dimensionality of the input tensor, `bond_dim` is the bond dimension of the TN layer, `activation` is the activation function to use, `kernel_initializer` is the initializer for the weight matrices, `use_bias` specifies whether to include a bias term, and `bias_initializer` is the initializer for the bias term.

3. Create an instance of the TNModel class:
   ```python
   tn_model = TNModel(num_layers, MPO_units, output_dim, bond_dim, activation, use_bias, kernel_initializer, bias_initializer, dif_equation)
   ```

   Replace the arguments with the desired values. `num_layers` is the number of TN layers, `MPO_units` is the number of units in the MPO tensor, `output_dim` is the dimension of the output, `bond_dim` is the bond dimension of the TN layer (optional), `activation` is the activation function to use, `use_bias` specifies whether to include a bias term, `kernel_initializer` is the initializer for the weight matrices, `bias_initializer` is the initializer for the bias term, and `dif_equation` is a callable representing the one-dimensional fourth-order PDE.

4. Use the TNModel to perform forward pass and compute the PDE loss:
   ```python
   y_pred = tn_model.call(x)
   pde_loss = tn_model.compute_pde_loss(x)
   ```

   Replace `x` with the input tensor.

## :rocket: Examples
An example usage of TPINNs can be found in the `examples` directory. It demonstrates how to solve a physics-based problem using TPINNs.

## :busts_in_silhouette: Contributing
Contributions to this repository are welcome. Feel free to open issues or submit pull requests.

## :page_with_curl: License
This project is licensed under the [MIT License](LICENSE).

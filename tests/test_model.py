import pytest
import tensorflow as tf
import numpy as np
from model import TNLayer, TNModel


def test_tnlayer_output_shape():
    # Create an instance of TNLayer with different keyword arguments
    input_dim = 16
    batch_size = 5
    input_shape = (batch_size, input_dim)

    bond_dim = 8
    activation = "sigmoid"
    kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
    use_bias = False
    bias_initializer = "ones"
    use_batch_norm = True
    batch_norm_momentum = 0.95

    tn_layer = TNLayer(
        input_dim=input_dim,
        bond_dim=bond_dim,
        activation=activation,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
        bias_initializer=bias_initializer,
        use_batch_norm=use_batch_norm,
        batch_norm_momentum=batch_norm_momentum,
    )

    input_batch = np.random.randn(*input_shape)
    input_tensor = tf.convert_to_tensor(input_batch, dtype=tf.float32)

    # Call the TNLayer on the input tensor
    output_tensor = tn_layer(input_tensor)

    # Check the output shape
    expected_output_shape = (batch_size, input_dim)
    assert output_tensor.shape == expected_output_shape


def test_forward_pass():
    output_dim = 3
    model = TNModel(
        MPO_units=16,
        num_layers=2,
        bond_dim=2,
        output_dim=output_dim,
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
    )

    input_dim = 1
    batch_size = 16
    input_batch = np.random.randn(batch_size, input_dim)

    # Convert the batch tensor to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_batch, dtype=tf.float32)

    # Perform forward pass
    output = model(input_tensor)

    # Ensure the output shape is [batch_size, output_dim]
    assert output.shape == (batch_size, output_dim)

    # Ensure the output tensor is of type float32
    assert output.dtype == tf.float32

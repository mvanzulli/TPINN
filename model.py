import tensorflow as tf
import numpy as np


class TNLayer(tf.keras.layers.Layer):
    """
    Tensor Network Layer

    This layer implements a Tensor Network layer in TensorFlow. The input dimension must be
    [16,64,256...] a perfect square. The layer contracts the MPO tensors A and B along the
    bond index and reshapes the resulting rank-4 tensor into a matrix.

    Args:
        input_dim (int): The dimensionality of the input tensor
        bond_dim (int, optional): The bond dimension of the Tensor Network layer. If not provided, it is set to d^2 // 2, where d is the square root of input_dim. Default is None.
        activation (str or callable, optional): Activation function to use. Must be a valid string identifier or a callable. Default is "relu".
        kernel_initializer (str or callable, optional): Initializer for the weight matrices A and B. Must be a valid string identifier or a callable. Default is "glorot_uniform".
        use_bias (bool, optional): Whether to include a bias term in the layer. Default is True.
        bias_initializer (str or callable, optional): Initializer for the bias term. Must be a valid string identifier or a callable. Default is "zeros".

    Raises:
        AssertionError: If input_dim is not a perfect square or bond_dim is not an integer.

    Attributes:
        d (int): The square root of input_dim.
        bond_dim (int): The bond dimension of the Tensor Network layer.
        activation (callable): The activation function.
        kernel_initializer (callable): The initializer for the weight matrices A and B.
        use_bias (bool): Whether to include a bias term in the layer.
        bias_initializer (callable): The initializer for the bias term.
        A (tf.Variable): The weight matrix A.
        B (tf.Variable): The weight matrix B.

    """

    def __init__(
        self,
        input_dim: int,
        bond_dim: int = None,
        activation: str = "relu",
        kernel_initializer: str = "glorot_uniform",
        use_bias: bool = True,
        bias_initializer: str = "zeros",
        use_batch_norm: bool = False,
        batch_norm_momentum: float = 0.99,
        **kwargs,
    ) -> None:
        super(TNLayer, self).__init__(**kwargs)

        assert np.sqrt(input_dim).is_integer()
        self.d = int(tf.math.sqrt(float(input_dim)))

        if not bond_dim:
            bond_dim = self.d**2 // 2
            assert bond_dim % 1 == 0, "bond_dim must be an integer"

        self.bond_dim = bond_dim
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.batch_norm_momentum = batch_norm_momentum
            self.batch_norm = tf.keras.layers.BatchNormalization(
                momentum=self.batch_norm_momentum
            )

    def build(self, input_shape: tuple) -> None:
        """
        Build the layer by initializing the weight matrices A and B.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            None
        """
        d = self.d
        self.RL = self.add_weight(
            shape=(self.bond_dim, d, d),
            initializer=self.kernel_initializer,
            trainable=True,
            name="RL",
        )

        self.RR = self.add_weight(
            shape=(self.bond_dim, d, d),
            initializer=self.kernel_initializer,
            trainable=True,
            name="RR",
        )

        super(TNLayer, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the forward pass of the layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor.
        """
        contracted_tensor = tf.einsum("abc,ade->bcde", self.RL, self.RR)

        reshaped_matrix = tf.reshape(
            contracted_tensor,
            (self.d**2, self.d**2),
        )

        output_tensor = tf.einsum("ab,cb->cb", reshaped_matrix, inputs)

        output_tensor = self.activation(output_tensor)

        return output_tensor


class TNModel(tf.keras.Model):
    """
    Tensor Network Model class.

    Attributes:
        num_layers (int): Number of TN layers.
        MPO_units (int): Number of units in the MPO tensor.
        output_dim (int): Dimension of the output.
        activation (str): Activation function to use (default is "tanh").
        use_bias (bool): Whether to use bias in the layers (default is True).
        kernel_initializer (str): Initializer for the kernel weights (default is "glorot_uniform").
        bias_initializer (str): Initializer for the bias weights (default is "zeros").

    Methods:
        call(x: tf.Tensor) -> tf.Tensor: Forward pass of the model.
    """

    def __init__(
        self,
        num_layers: int,
        MPO_units: int,
        output_dim: int,
        bond_dim: int = None,
        activation: str = "tanh",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        """
        Initialize TNModel with the given parameters.

        Args:
            num_layers (int): Number of TN layers.
            MPO_units (int): Number of units in the MPO tensor.
            output_dim (int): Dimension of the output.
            activation (str): Activation function to use (default is "tanh").
            use_bias (bool): Whether to use bias in the layers (default is True).
            kernel_initializer (str): Initializer for the kernel weights (default is "glorot_uniform").
            bias_initializer (str): Initializer for the bias weights (default is "zeros").
        """
        super(TNModel, self).__init__()
        self.num_layers = num_layers
        self.MPO_units = MPO_units
        self.bond_dim = bond_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.d = int(tf.math.sqrt(float(MPO_units)))

        self.dense_input = tf.keras.layers.Dense(
            MPO_units,
            activation="linear",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            trainable=False,
        )
        self.tn_layers = [
            TNLayer(
                input_dim=MPO_units,
                activation=self.activation,
                bond_dim=bond_dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                trainable=True,
            )
            for _ in range(num_layers)
        ]
        self.dense_output = tf.keras.layers.Dense(
            output_dim,
            activation="linear",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            trainable=True,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Perform a forward pass of the model.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        a = self.dense_input(x)
        for tn_layer in self.tn_layers:
            a = tn_layer(a)
        y = self.dense_output(a)
        return y

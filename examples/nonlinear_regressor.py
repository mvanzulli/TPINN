import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import TNModel

# Generate training data
f = np.sin
X_train = np.random.uniform(-3, 3, size=(1000, 1))  # Generate 1000 random values for X
Y_train = f(X_train) + np.random.normal(
    0, 0.1, size=(1000, 1)
)  # Add noise to the Y_train values

# Generate test data
X_test = np.random.uniform(-3, 3, size=(2000, 1))  # Generate 100 random values for X
Y_test = f(X_test)  # Calculate the corresponding values for Y using the function f

# Create the model
model = TNModel(MPO_units=64, bond_dim=2, num_layers=2, output_dim=1)

# Build the model
batch_input_shape = (1, 1)
model.build(batch_input_shape)
num_params = model.count_params()
print("Number of parameters:", num_params)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")

# Train the model and record the loss history
history = model.fit(
    X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test)
)

# Plot the actual data and the model predictions
X_range = np.linspace(-3, 3, 1000).reshape(-1, 1)
Y_pred = model.predict(X_range)

plt.scatter(X_train, Y_train, label="Actual")
plt.plot(X_range, Y_pred, color="red", label="Predicted")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

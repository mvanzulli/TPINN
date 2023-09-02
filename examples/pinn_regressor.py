import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import TNModel


# Define the function to approximate
f = np.sin


def pde(x: tf.Tensor, y: tf.Tensor, dy_dx: tf.Tensor):
    return dy_dx - tf.cos(x)


# Generate training data with noise
a = 5
x_min = -a
x_max = a
num_samples_train = 1000
X_train = np.random.uniform(x_min, x_max, size=(num_samples_train, 1))
std_dev_noise = 0.1
Y_train = f(X_train) + np.random.normal(0, std_dev_noise, size=(num_samples_train, 1))

# Generate test data
num_samples_test = 3000
X_test = np.random.uniform(x_min, x_max, size=(num_samples_test, 1))
Y_test = f(X_test)

# Generate offset data
offset = 2
num_offset_samples = 1000
X_range_offset = np.linspace(
    x_min - offset, x_max + offset, num_offset_samples
).reshape(-1, 1)

# Create arrays to store experiment results
train_loss_history = []
val_loss_history = []
Y_pred_test_history = []
Y_pred_offset_history = []

# Train parameters
num_runs = 1
num_epochs = 10_000
batch_size = 64

# Model hyper-parameters
bond_dim = 2
num_layers = 2

for run in range(num_runs):
    pinn_model = TNModel(
        MPO_units=16,
        bond_dim=bond_dim,
        num_layers=num_layers,
        output_dim=1,
        dif_equation=pde,
    )

    # Print model num of parameters
    pinn_model.build(input_shape=(batch_size, 1))

    if run == 0:
        num_params = pinn_model.count_params()

    def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        pde_loss = tf.cast(pinn_model.compute_pde_loss(X_train), dtype=tf.float32)
        data_loss = tf.reduce_mean(tf.square(y_pred - y_true))
        total_loss = data_loss + pde_loss
        return total_loss

    pinn_model.compile(optimizer="adam", loss=custom_loss)

    history = pinn_model.fit(
        X_train,
        Y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
    )

    train_loss_history.append(history.history["loss"])
    val_loss_history.append(history.history["val_loss"])
    Y_pred_offset = pinn_model.predict(X_range_offset)
    Y_pred_offset_history.append(Y_pred_offset)

print("Number of parameters:", num_params)


# Compute mean and standard deviation of training and validation loss
mean_train_loss = np.mean(train_loss_history, axis=0)
std_train_loss = np.std(train_loss_history, axis=0)
mean_val_loss = np.mean(val_loss_history, axis=0)
std_val_loss = np.std(val_loss_history, axis=0)

mean_Y_pred_offset = np.mean(Y_pred_offset_history, axis=0)
std_Y_pred_offset = np.std(Y_pred_offset_history, axis=0)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot mean and standard deviation of training and validation loss
axes[0].plot(mean_train_loss, label="Training Loss")
axes[0].fill_between(
    range(len(mean_train_loss)),
    mean_train_loss - std_train_loss,
    mean_train_loss + std_train_loss,
    alpha=0.3,
    label=None,
)
axes[0].plot(mean_val_loss, label="Validation Loss")
axes[0].fill_between(
    range(len(mean_val_loss)),
    mean_val_loss - std_val_loss,
    mean_val_loss + std_val_loss,
    alpha=0.3,
    label=None,
)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
# axes[0].set_yscale("log")
axes[0].legend()
axes[0].grid()

# Plot offset predictions
axes[1].plot(
    X_range_offset, mean_Y_pred_offset, color="black", label="Offset prediction"
)
axes[1].fill_between(
    X_range_offset.flatten(),
    mean_Y_pred_offset.flatten() - std_Y_pred_offset.flatten(),
    mean_Y_pred_offset.flatten() + std_Y_pred_offset.flatten(),
    alpha=0.3,
    color="gray",
)
axes[1].scatter(X_train, Y_train, label="Training Data", color="red")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].legend()
axes[1].grid()

title = f"TNN-PINN Nonlinear Regressor: NP = {num_params}"
fig.suptitle(title, fontsize=16, ha="center")
plt.savefig("examples/pinn_regressor.png")

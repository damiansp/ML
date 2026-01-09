import time

from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch


def plot_results(model, distances, times):
    '''Plots the actual data points and the model's predicted line for a given
    dataset.
    Parameters:
    - model: The trained machine learning model to use for predictions.
    - distances: The input data points (features) for the model.
    - times: The target data points (labels) for the plot.
    '''
    model.eval()
    with torch.no_grad():
        predicted_times = model(distances)
    plt.figure(figsize=(8, 6))
    plt.plot(
        distances.numpy(),
        times.numpy(),
        color='orange',
        marker='o',
        linestyle='None',
        label='Ground Truth')
    plt.plot(
        distances.numpy(),
        predicted_times.numpy(),
        color='green',
        marker='None',
        label='Predictions')
    plt.title('Model Performance')
    plt.xlabel('Distance (mi)')
    plt.ylabel('Time (min)')
    plt.legend()
    plt.show()


def plot_nonlinear_comparison(model, new_distances, new_times):
    '''Compares and plots the predictions of a model against new, non-linear
    data.
    Parameters:
    - model: The trained model to be evaluated.
    - new_distances: The new input data for generating predictions.
    - new_times: The actual target values for comparison.
    '''
    model.eval()
    with torch.no_grad():
        preds = model(new_distances)
    plt.figure(figsize=(8, 6))
    plt.plot(
        new_distances.numpy(),
        new_times.numpy(),
        color='orange',
        marker='o',
        linestyle='None',
        label='Ground Truth')
    plt.plot(
        new_distances.numpy(),
        preds.numpy(),
        color='green',
        marker='None',
        label='Predictions')
    plt.title('Model vs Non-linear Reality')
    plt.xlabel('Distance (mi)')
    plt.ylabel('Time (min)')
    plt.legend()
    plt.show()


def plot_data(distances, times, normalize=False):
    '''Creates a scatter plot of the data points.
    Parameters:
    - distances: The input data points for the x-axis.
    - times: The target data points for the y-axis.
    0 normalize: A boolean flag indicating whether the data is normalized.
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(
        distances.numpy(),
        times.numpy(),
        color='orange',
        marker='o',
        linestyle='none',
        label='Actual Delivery Times')
    if normalize:
        plt.title('Normalized Delivery Data (Bikes & Cars)')
        plt.xlabel('Normalized Distance')
        plt.ylabel('Normalized Time')
    else:
        plt.title('Delivery Data (Bikes & Cars)')
        plt.xlabel('Distance (miles)')
        plt.ylabel('Time (minutes)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_final_fit(
        model, distances, times, distances_norm, times_std, times_mean):
    '''Plots the predictions of a trained model against the original data,
    after de-normalizing the predictions.
    Parameters:
    - model: The trained model used for prediction.
    - distances: The original, un-normalized input data.
    - times: The original, un-normalized target data.
    - distances_norm: The normalized input data for the model.
    - times_std: The standard deviation used for de-normalization.
    - times_mean: The mean value used for de-normalization.
    '''
    model.eval()
    with torch.no_grad():
        predicted_norm = model(distances_norm)
    predicted_times = (predicted_norm * times_std) + times_mean
    plt.figure(figsize=(8, 6))
    plt.plot(
        distances.numpy(),
        times.numpy(),
        color='orange',
        marker='o',
        linestyle='none',
        label='Actual Data (Bikes & Cars)')
    plt.plot(
        distances.numpy(),
        predicted_times.numpy(),
        color='green',
        label='Non-Linear Model Predictions')
    plt.title('Non-Linear Model Fit vs. Actual Data')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Time (minutes)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_progress(epoch, loss, model, distances_norm, times_norm):
    '''Plots the training progress of a model on normalized data,
    showing the current fit at each epoch.
    Parmeters:
    - epoch: The current training epoch number.
    - loss: The loss value at the current epoch.
    - model: The model being trained.
    - distances_norm: The normalized input data.
    - times_norm: The normalized target data.
    '''
    clear_output(wait=True)
    predicted_norm = model(distances_norm)
    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()
    y_pred_plot = predicted_norm.detach().numpy()
    sorted_indices = x_plot.argsort(axis=0).flatten()
    plt.figure(figsize=(8, 6))
    plt.plot(
        x_plot,
        y_plot,
        color='orange',
        marker='o',
        linestyle='none',
        label='Actual Normalized Data')
    plt.plot(
        x_plot[sorted_indices],
        y_pred_plot[sorted_indices],
        color='green',
        label='Model Predictions')
    plt.title(f'Epoch: {epoch + 1} | Normalized Training Progress')
    plt.xlabel('Normalized Distance')
    plt.ylabel('Normalized Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Pause briefly to allow the plot to be rendered
    time.sleep(0.05)

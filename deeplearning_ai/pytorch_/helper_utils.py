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

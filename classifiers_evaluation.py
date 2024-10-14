from matplotlib import pyplot as plt
from classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(fit: Perceptron, _, __):
            losses.append(fit.loss(X, y))

        model = Perceptron(callback=loss_callback)
        model.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses, label='Training Loss', color='blue', linestyle='-', marker='o')
        plt.title(f"Perceptron Training Progress - {n} Data")
        plt.xlabel("Iteration Number")
        plt.ylabel("Misclassification Loss")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(f"Perceptron Training - {n.replace(' ', '_').lower()}.png")
        plt.close()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes().fit(X, y)
        lda = LDA().fit(X, y)
        y_gnb = gnb.predict(X)
        y_lda = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from loss_functions import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                rf"$\text{{Gaussian Naive Bayes (accuracy={round(100 * accuracy(y, y_gnb), 2)}%)}}$",
                                rf"$\text{{Linear Discriminant Analysis (accuracy={round(100 * accuracy(y, y_lda), )}%)}}$"))

        # Add traces for data-points setting symbols and colors
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=y_gnb, symbol=class_symbols[y], colorscale=class_colors(3))),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=y_lda, symbol=class_symbols[y], colorscale=class_colors(3)))],
                       rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers",
                                   marker=dict(symbol="x", color="black", size=15)),
                        go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                                   marker=dict(symbol="x", color="black", size=15))],
                       rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for class_index in range(3):
            # Get ellipses for the Gaussian Naive Bayes and Linear Discriminant Analysis models
            gnb_ellipse = get_ellipse(gnb.mu_[class_index], np.diag(gnb.vars_[class_index]))
            lda_ellipse = get_ellipse(lda.mu_[class_index], lda.cov_)

            # Add ellipses to the figure
            fig.add_traces([gnb_ellipse, lda_ellipse], rows=[1, 1], cols=[1, 2])

        # Update y-axis to have the same scale as the x-axis
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Set subplot titles
        fig.update_layout(
            title_text=f"Comparison of Gaussian Classifiers - {f[:-4]}.npy Dataset",
            title_x=0.5,
            title_y=0.95,
            title_font=dict(size=18),
            annotations=[
                dict(
                    text=f"Gaussian Naive Bayes (Accuracy = {round(100 * accuracy(y, y_gnb), 2):.2f}%)",
                    x=0.2,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                ),
                dict(
                    text=f"Linear Discriminant Analysis (Accuracy = {round(100 * accuracy(y, y_lda), ):.2f}%)",
                    x=0.8,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ],
            width=1000,
            height=500,
            showlegend=False
        )

        html_filename = f"Naive Bayes VS LDA - {f[:-4]}.html"

        fig.write_html(html_filename)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

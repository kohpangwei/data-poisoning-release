import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse


class Plotter(object):
    def __init__(self, X, axes=None, figsize=None, padding=1, axis_equal=False, fig=None, ax=None):
        sns.set(font_scale=1.5)
        sns.set_style(style='white')
        # sns.set_style({'font.sans-serif': [u'Droid Sans']})
        if figsize is None:
            figsize = (10, 8)
        if axes is None:
            axes = np.zeros((2, X.shape[1]))
            axes[0, 0] = 1
            axes[1, 1] = 1
        
        # Normalize each axis to have unit l2 norm    
        axes = axes / np.reshape(np.linalg.norm(axes, axis=1), (-1, 1))

        # Orthogonalize 2nd axis wrt 1st axis
        axes[1, :] = axes[1, :] - axes[1, :].dot(axes[0, :]) * axes[0, :]
        axes[1, :] = axes[1, :] / np.linalg.norm(axes[1, :])

        self.axes = axes
        
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(figsize=figsize)

        self.x_min, self.x_max, self.y_min, self.y_max = self.get_boundaries(X, padding=padding)

        if axis_equal:
            self.x_min = np.min((self.x_min, self.y_min))
            self.y_min = np.min((self.x_min, self.y_min))
            self.x_max = np.min((self.x_max, self.y_max))
            self.y_max = np.min((self.x_max, self.y_max))

        ax.set(xlim=(self.x_min, self.x_max), ylim=(self.y_min, self.y_max))

        self.fig = fig
        self.ax = ax

    def proj(self, X, axis=None):
        if axis is None:
            return X.dot(self.axes.T)
        else:
            return X.dot(self.axes[axis, :])
    
    def get_boundaries(self, X, padding=1):
        x_min = np.min(self.proj(X, 0)) - padding
        x_max = np.max(self.proj(X, 0)) + padding
        y_min = np.min(self.proj(X, 1)) - padding
        y_max = np.max(self.proj(X, 1)) + padding
        return x_min, x_max, y_min, y_max

    def plot_points_train(self, X, Y, linewidth=1, alpha=0.4, subsample=None):       
        if subsample:
            idx_to_sample = np.random.choice(
                X.shape[0], 
                size=int(subsample * X.shape[0]),
                replace=False)
        else:
            idx_to_sample = slice(0, X.shape[0])

        colors = ['red' if y == 1 else 'blue' for y in Y[idx_to_sample]]
        self.ax.scatter(
            self.proj(X[idx_to_sample, :], 0),
            self.proj(X[idx_to_sample, :], 1),
            s=80, c=colors, edgecolor="white", linewidth=linewidth, alpha=alpha)
    
    def plot_points_poison(self, X, Y, linewidth=2, alpha=0.4, marker='o', edgecolor='black', 
                           poscolor='red', negcolor='blue', size=80, linestyle='-', label=None,
                           zorder=4):    
        colors = [poscolor if y == 1 else negcolor for y in Y]
        self.ax.scatter(
            self.proj(X, 0),
            self.proj(X, 1),
            s=size, c=colors, linewidth=linewidth, linestyle=linestyle, alpha=alpha, marker=marker, edgecolor=edgecolor, label=label,
            zorder=zorder)

    def plot_points_manual(self, X, color):
        self.ax.scatter(
            self.proj(X, 0),
            self.proj(X, 1),
            s=80, c=color, edgecolor="white", linewidth=1, alpha=0.7)    
    
    def draw_decision_boundary(self, params, bias):
        # This assumes that the second axis is the orthogonal complement of w (params) wrt v 
        # Probably should refactor this class to make this assumption explicit
        y0 = (-self.x_min * params.dot(self.axes[0, :]) - bias) / params.dot(self.axes[1, :])
        y1 = (-self.x_max * params.dot(self.axes[0, :]) - bias) / params.dot(self.axes[1, :])
        self.draw_line(y0=y0, y1=y1)

    def draw_line(self, y0=None, y1=None, x0=None, x1=None, color=None, alpha=1, linestyle='-', linewidth=1):
        if y0 is None: y0 = self.y_min
        if y1 is None: y1 = self.y_max
        if x0 is None: x0 = self.x_min
        if x1 is None: x1 = self.x_max

        if color is None:
            self.ax.plot([x0, x1], [y0, y1], alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        else:
            self.ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)

    def draw_plane(self, v, mu, r, color=None):
        """
        Draws line corresponding to v^T x = r
        """
        v0 = v.dot(self.axes[0, :])
        v1 = v.dot(self.axes[1, :])

        y_left = (r + v.dot(mu) - self.x_min * v0) / v1
        y_right = (r + v.dot(mu) - self.x_max * v0) / v1
        print(y_left, y_right)
        self.draw_line(y0=y_left, y1=y_right, color=color)
    

def plot_flat_bwimage(X, y=None, side=28):
    X = np.reshape(X, (side, side)).T
        
    with sns.axes_style("white"):
        if y is not None:
            plt.title('Label is %s' % y)
        plt.imshow(X, cmap='gray', interpolation='none')


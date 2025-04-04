import numpy as np
import matplotlib.pyplot as plt
import queue
import threading
import time
import math
from matplotlib.widgets import Slider

# This class handles the plotting of clusters in a separate thread
# It uses matplotlib to create a live plot of the clusters
# It uses a queue to receive data from the main thread

class ClusterHandler:
    def __init__(self, xaxsis=1000, yaxsis=1000):
        self.plot_queue = queue.Queue()
        self.xaxsis = xaxsis
        self.yaxsis = yaxsis
        self.running = True
        # Start the plotting thread
        #self.plot_thread = threading.Thread(target=self.plot_loop)
        #self.plot_thread.daemon = True  # Daemon thread so it closes with the main program
        #self.plot_thread.start()

    def update(self): self.plot_loop()

    def plot_loop(self):
        """
        This loop runs in a separate thread, waiting for new data to update the plot.
        """
        #plt.ion()  # Turn on interactive mode
        #while self.running:
        try:
            # Wait for new plot data (with a short timeout to allow a graceful shutdown)
            points_np, labels, kalman_state = self.plot_queue.get(timeout=1)
            self.live_plot_clusters(points_np, labels, kalman_state)
        except queue.Empty as e:
            pass
    def live_plot_clusters(self, points_np, labels, kalman_state=None):
        """
        Plot clusters and optionally the Kalman filter estimate.
        """
        plt.clf()  # Clear the figure
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            marker = 'o'
            if k == -1:
                marker = 'x'
                col = 'k'
            class_member_mask = (labels == k)
            xy = points_np[class_member_mask]
            if xy.size > 0:
                plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=col, markersize=6, linestyle='None')
        
        # Mark the sensor at (0,0)
        plt.plot(0, 0, 'r*', markersize=15, label="Sensor (0,0)")
        
        # Plot the Kalman filter estimate if available
        if kalman_state is not None:
            plt.plot(kalman_state[0], kalman_state[1], 'bo', markersize=10, label="Kalman Estimate")
        
        # ----------------------------------
        # Interactive distance modifier
        # ----------------------------------


        # ----------------------------------
        # None
        # ----------------------------------
        plt.title("Live Clusters")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.xlim(-self.xaxsis, self.xaxsis)
        plt.ylim(-self.yaxsis, self.yaxsis)
        plt.legend()
        plt.draw()
        plt.pause(0.001)
    
    def update_plot(self, points_np, labels, kalman_state=None):
        self.plot_queue.put((points_np, labels, kalman_state))
    
    def stop(self):
        self.running = False
        self.plot_thread.join()

    @staticmethod
    def compute_cluster_means(points_np, labels):
        cluster_means = {}
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # ignore noise
                continue
            cluster_points = points_np[labels == label]
            mean_position = np.mean(cluster_points, axis=0)
            cluster_means[label] = mean_position
        return cluster_means
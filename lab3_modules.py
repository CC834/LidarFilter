import threading
import time
from ydlidar_x2 import YDLidarX2
import math
import numpy as np
import json
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import queue
Task = "Task1"

class lidarModules():
    def __init__(self,lidar, CH):
        self.lidar: YDLidarX2 = lidar
        self.clusterHandler: ClusterHandler = CH

        self.scanStatus: bool = False
        self.GUI_PLOT = True
        self.debugInfo = False

        self.angle0: float = 45
        self.angle1: float = 360-40

        # Settings for DBSCAN
        self.eps=80 # It controls the local neighborhood of the points.
        self.min_samples=5 # min_samples primarily controls how tolerant the algorithm is towards nois

        if self.GUI_PLOT: self.plotter = self.clusterHandler(xaxsis=1500, yaxsis=1500)
        pass
    
    def runScan(self):
        pass

    
    def scan(self):
        while self.scanStatus:
            if self.lidar.available:
                distances = self.lidar.get_data()  # Returns 360 distance values in mm
                points = []
                for angle, dist in enumerate(distances):
                    if angle <= self.angle0 or angle >= self.angle1:
                        if dist <= 0: continue

                        radian = math.radians(angle)
                        x = dist * math.cos(radian)
                        y = dist * math.sin(radian)
                        points.append([x, y])

                if points:
                    points_np = np.array(points)
                    clustering = DBSCAN(eps=80, min_samples=5).fit(points_np)
                    labels = clustering.labels_
                    #live_plot_clusters(points_np, labels, 1000, 1000)
                    
                    match Task:
                        case "Task1":
                            self.plotter.update_plot(points_np, labels)
                        case "Task2":
                            pass
                        case "Task3":
                            pass
                        case "task3":
                            pass
                        
                    cluster_means = self.clusterHandler.compute_cluster_means(points_np, labels)
                    for label, mean in cluster_means.items():
                        if self.debugInfo: print(f"Cluster {label}: Mean position (mm): {mean}")    
                    
    def stopScan(self): self.scanStatus = False
    def startScan(self): self.scanStatus = True

class ClusterHandler:
    def __init__(self, xaxsis=1000, yaxsis=1000):
        self.plot_queue = queue.Queue()
        self.xaxsis = xaxsis
        self.yaxsis = yaxsis
        self.running = True
        # Start the plotting thread
        self.plot_thread = threading.Thread(target=self.plot_loop)
        self.plot_thread.daemon = True  # Daemon thread so it closes with the main program
        self.plot_thread.start()

    def plot_loop(self):
        """
        This loop runs in a separate thread, waiting for new data to update the plot.
        """
        plt.ion()  # Turn on interactive mode
        while self.running:
            try:
                # Wait for new plot data (with a short timeout to allow a graceful shutdown)
                points_np, labels, kalman_state = self.plot_queue.get(timeout=0.05)
                self.live_plot_clusters(points_np, labels, kalman_state)
            except queue.Empty:
                continue

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
        
        plt.title("Live Clusters")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.xlim(-self.xaxsis, self.xaxsis)
        plt.ylim(-self.yaxsis, self.yaxsis)
        plt.legend()
        plt.draw()
        plt.pause(0.001)
    
    def getCluster(self, points):
                if points:
                    points_np = np.array(points)
                    clustering = DBSCAN(eps=self.eps, min_samples=self.min_sample).fit(points_np)
                    labels = clustering.labels_

                    #live_plot_clusters(points_np, labels, 1000, 1000)

                    return
                    cluster_means = compute_cluster_means(points_np, labels)
                    for label, mean in cluster_means.items():
                        if debugInfo: print(f"Cluster {label}: Mean position (mm): {mean}")
    
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
    
    def update_plot(self, points_np, labels, kalman_state=None):
        self.plot_queue.put((points_np, labels, kalman_state)) 
    
    def stop(self):
        self.running = False
        self.plot_thread.join()
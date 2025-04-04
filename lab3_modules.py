import threading
import time
from ydlidar_x2 import YDLidarX2
import math
import numpy as np
import json
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import queue
import warnings
warnings.filterwarnings("ignore", "Starting a Matplotlib GUI outside of the main thread", UserWarning)
from kalmanFilter import kalman_filter
import traceback

TASK = "Task3"

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


class lidarModules:
    def __init__(self, lidar, xaxsis=1000, yaxsis=1000):
        self._lidar: YDLidarX2 = lidar
        self.scanStatus: bool = False
        self.GUI_PLOT = True
        self.debugInfo = True

        self.angle0: float = 45
        self.angle1: float = 360 - 40

        # Settings for DBSCAN
        self.eps = 80  # It controls the local neighborhood of the points.
        self.min_samples = 5  # min_samples controls tolerance towards noise

        if self.GUI_PLOT:
            self.plotter = ClusterHandler(xaxsis=1500, yaxsis=1500)

        if TASK == "Task1": pass
        if TASK == "Task2": pass
        if TASK == "Task3": self.kf = kalman_filter(lidar)
        if TASK == "Task4": pass


    def runScan(self):
        self.scanStatus = True
        try: 
            self._scan_thread = threading.Thread(target=self._scan)
            self._scan_thread.daemon = True  # Daemon thread so it closes with the main program
            self._scan_thread.start()
        except KeyboardInterrupt:
            pass
        # Removed the line that sets self.scanStatus = False here.
    
    def _scan(self):
        while self.scanStatus:
            if self._lidar.available:
                distances = self._lidar.get_data()  # Use self._lidar consistently
                points = []
                for angle, dist in enumerate(distances):
                    if angle <= self.angle0 or angle >= self.angle1:
                        if dist <= 0:
                            continue
                        radian = math.radians(angle)
                        x = dist * math.cos(radian)
                        y = dist * math.sin(radian)
                        points.append([x, y])

                if points:
                    points_np = np.array(points)
                    clustering = DBSCAN(eps=80, min_samples=5).fit(points_np)
                    labels = clustering.labels_
                    cluster_means = self.plotter.compute_cluster_means(points_np, labels)
                    match TASK:
                        case "Task1":
                            self.plotter.update_plot(points_np, labels)
                        case "Task2":
                            pass
                        case "Task3":
                            self.plotter.update_plot(points_np, labels, kalman_state=self.kf.X)
                            if not cluster_means: return
                            selected_label, selected_mean = min(cluster_means.items(), key=lambda item: np.linalg.norm(item[1])) 
                            if self.debugInfo: print(f"Selected Cluster {selected_label}: Mean position (mm): {selected_mean}")
                            # Form the observation vector: use [x, y, 0, 0] (velocity unobserved)
                            z = np.array([selected_mean[0], selected_mean[1], 0, 0])
                            self.kf.predict()
                            self.kf.update(z)          
                            if self.debugInfo: print(f"Kalman filter state: {self.kf.X}")
                        case "task3":
                            pass

                    
                    for label, mean in cluster_means.items():
                        if self.debugInfo:
                            print(f"Cluster {label}: Mean position (mm): {mean}")    
            time.sleep(0.1)
    
    def stopScan(self):
        self.scanStatus = False
    
    def _startScan(self):
        self.scanStatus = True

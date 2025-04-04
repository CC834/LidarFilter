# In task1_module.py

import time
import math
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import json

debug = False
debugInfo = False

#
# Task 1: Detection Algorithm
#

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

def live_plot_clusters(points_np, labels, xaxsis=1000, yaxsis=100, kalman_state=None):
    """
    Plots clusters and optionally the Kalman filter estimate.
    """
    plt.clf()
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            marker = 'x'
            col = 'k'
        else:
            marker = 'o'
        class_member_mask = (labels == k)
        xy = points_np[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=col, markersize=6, linestyle='None')
    
    # Mark the sensor at (0,0)
    plt.plot(0, 0, 'r*', markersize=15, label="Sensor (0,0)")
    
    # Plot the Kalman filter estimate if available
    if kalman_state is not None:
        plt.plot(kalman_state[0], kalman_state[1], 'bo', markersize=10, label="Kalman Estimate")
    
    plt.title("Live Clusters")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.xlim(-xaxsis, xaxsis)
    plt.ylim(-yaxsis, yaxsis)
    plt.legend()
    plt.draw()
    plt.pause(0.001)





def runScanCluster(lidar):
    try:
        while True:
            if lidar.available:
                distances = lidar.get_data()  # Returns 360 distance values in mm
                points = []
                for angle, dist in enumerate(distances):
                    if angle <= 45 or angle >= 360-40:
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
                    live_plot_clusters(points_np, labels, 1000, 1000)
                    cluster_means = compute_cluster_means(points_np, labels)
                    for label, mean in cluster_means.items():
                        if debugInfo: print(f"Cluster {label}: Mean position (mm): {mean}")  
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


#
# Task 2: Record Datasets (omitted here for brevity)
#

#
# Task 3: Kalman Filter - Brownian Motion
#

class kalman_filter:
    def __init__(self, lidar, par = 2):
        # Kalman Filter parameters
        n = 4  # State: [x, y, vx, vy]
        self.lidar = lidar
        np.random.seed(42)
        self.X = np.array([0, 0, 0, 0], dtype=float)  # Initial state
        match par:
            case 0:
                self.P = np.eye(n) * 500                      # High initial uncertainty
                self.Q = np.eye(n) * 2                      # Process noise covariance
                self.R = np.eye(n) * 50                       # Observation noise covariance
            case 1:
                self.P = np.eye(n) * 500                      # Start with high uncertainty.
                self.Q = np.eye(n) * 5                        # Increase process noise for faster adaptation.
                self.R = np.eye(n) * 25  
            case 2:
                self.P = np.eye(n) * 300                      # Slightly lower initial uncertainty.
                self.Q = np.eye(n) * 8                        # Higher process noise for even more responsiveness.
                self.R = np.eye(n) * 30 

    def predict(self):
        # For Brownian motion, the state transition matrix is identity
        # Thus, the prediction step is:
        # X_pred = X (unchanged) and P_pred = P + Q
        self.P = self.P + self.Q
        return self.X, self.P

    def update(self, z):
        # With observation matrix H = I, the update equations are:
        # y = z - X
        # S = P + R
        # K = P * inv(S)
        # X = X + K * y
        # P = (I - K) * P
        y = z - self.X
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(4) - K) @ self.P
        return self.X, self.P

def runScanClusterKF(lidar):
    """
    This function integrates the clustering function with the Kalman filter.
    It processes a scan, selects one cluster (the one closest to the sensor),
    and uses its mean as the observation to update the Kalman filter.
    """
    kf = kalman_filter(lidar)
    try:
        while True:
            if lidar.available:
                distances = lidar.get_data()  # Returns 360 distance values in mm
                points = []
                for angle, dist in enumerate(distances):
                    if angle <= 45 or angle >= 360-40:
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
                    
                    live_plot_clusters(points_np, labels, 1500, 1500, kalman_state=kf.X)
                    cluster_means = compute_cluster_means(points_np, labels)
                    if cluster_means:
                        # Select the cluster closest to (0,0)
                        selected_label, selected_mean = min(
                            cluster_means.items(),
                            key=lambda item: np.linalg.norm(item[1])
                        )
                        if debugInfo: print(f"Selected Cluster {selected_label}: Mean position (mm): {selected_mean}")
                        # Form the observation vector: use [x, y, 0, 0] (velocity unobserved)
                        z = np.array([selected_mean[0], selected_mean[1], 0, 0])
                        kf.predict()
                        kf.update(z)
                        if debugInfo: print(f"Kalman filter state: {kf.X}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


#
# Task 4: Kalman Filter – Constant Velocity Model
#

class kalman_filter_cv:
    def __init__(self, lidar, dt=0.1):
        """
        Constant velocity model Kalman filter.
        State vector: [x, y, vx, vy]
        dt: Time step between updates.
        """
        n = 4
        self.lidar = lidar
        self.dt = dt
        np.random.seed(42)
        self.X = np.array([0, 0, 0, 0], dtype=float)  # Initial state
        self.P = np.eye(n) * 500                      # Initial state uncertainty
        self.Q = np.eye(n) * 2                        # Process noise covariance (tunable)
        self.R = np.eye(n) * 50                       # Measurement noise covariance
        # Constant velocity state transition matrix A:
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1,     0],
            [0, 0, 0,     1]
        ])

    def predict(self):
        # Predict next state based on constant velocity model.
        self.X = self.A @ self.X
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.X, self.P

    def update(self, z):
        # With observation matrix H = I (direct observation of full state)
        y = z - self.X
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(4) - K) @ self.P
        return self.X, self.P
    

def runScanClusterKF_CV(lidar, dt=0.1):
    """
    Run live clustering and update the constant velocity Kalman filter.
    This function is similar to your runScanClusterKF but uses the
    constant velocity model (kalman_filter_cv) for prediction.
    """
    kf_cv = kalman_filter_cv(lidar, dt)
    try:
        while True:
            if lidar.available:
                distances = lidar.get_data()  # 360 distance values in mm
                points = []
                # Process only points from specific angles
                for angle, dist in enumerate(distances):
                    if angle <= 45 or angle >= 360 - 40:
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
                    live_plot_clusters(points_np, labels, 1500, 1500, kalman_state=kf_cv.X)
                    cluster_means = compute_cluster_means(points_np, labels)
                    if cluster_means:
                        # Select the cluster closest to the sensor (0,0)
                        selected_label, selected_mean = min(
                            cluster_means.items(),
                            key=lambda item: np.linalg.norm(item[1])
                        )
                        print(f"Selected Cluster {selected_label}: Mean position (mm): {selected_mean}")
                        # Create an observation vector: [x, y, 0, 0] (velocity is unobserved)
                        z = np.array([selected_mean[0], selected_mean[1], 0, 0])
                        kf_cv.predict()
                        kf_cv.update(z)
                        print(f"Constant Velocity Kalman filter state: {kf_cv.X}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass



#
# Task 5: Noise Parameters and Missing Observations
#


#
# Task 5: Noise Parameters and Missing Observations
#

class kalman_filter_cv2:
    def __init__(self, lidar, dt=0.1, Q_factor=2):
        """
        Constant velocity model Kalman filter with tunable process noise.
        Q_factor is multiplied with the identity matrix to set Q.
        """
        n = 4
        self.lidar = lidar
        self.dt = dt
        np.random.seed(42)
        self.X = np.array([0, 0, 0, 0], dtype=float)  # Initial state
        self.P = np.eye(n) * 500                      # Initial state uncertainty
        self.Q = np.eye(n) * Q_factor                 # Process noise covariance (tune this)
        self.R = np.eye(n) * 50                       # Measurement noise covariance
        # Constant velocity state transition matrix A:
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1,     0],
            [0, 0, 0,     1]
        ])

    def predict(self):
        # Predict next state using constant velocity model.
        self.X = self.A @ self.X
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.X, self.P

    def update(self, z):
        # With H = I (full observation)
        y = z - self.X
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(4) - K) @ self.P
        return self.X, self.P
    

def runScanClusterKF_CV_Missing(lidar, dt=0.1, Q_factor=2, missing_rate=0.2):
    """
    This function runs live clustering with the constant velocity Kalman filter.
    It simulates missing observations by skipping the update step at random.
    
    Parameters:
      dt: Time step for the filter.
      Q_factor: Scaling factor for process noise covariance Q.
      missing_rate: Probability (0 to 1) that a measurement is missing.
                    For example, 0.2 means 20% of the measurements are skipped.
    """
    kf_cv = kalman_filter_cv2(lidar, dt, Q_factor)
    try:
        while True:
            if lidar.available:
                distances = lidar.get_data()  # Returns 360 distance values in mm
                points = []
                # Process only points from specific angles (e.g., front view)
                for angle, dist in enumerate(distances):
                    if angle <= 45 or angle >= 360-40:
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
                    live_plot_clusters(points_np, labels, 1500, 1500, kalman_state=kf_cv.X)
                    cluster_means = compute_cluster_means(points_np, labels)
                    if cluster_means:
                        # Select the cluster closest to the sensor at (0,0)
                        selected_label, selected_mean = min(
                            cluster_means.items(),
                            key=lambda item: np.linalg.norm(item[1])
                        )
                        print(f"Selected Cluster {selected_label}: Mean position (mm): {selected_mean}")
                        # Form observation vector: [x, y, 0, 0]
                        z = np.array([selected_mean[0], selected_mean[1], 0, 0])
                        
                        # Always predict, but sometimes skip the update to simulate missing observations.
                        kf_cv.predict()
                        if np.random.rand() > missing_rate:
                            kf_cv.update(z)
                            print(f"Update with measurement: {kf_cv.X}")
                        else:
                            print(f"Missing measurement, prediction only: {kf_cv.X}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
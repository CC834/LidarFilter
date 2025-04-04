import threading
import time
from ydlidar_x2 import YDLidarX2
import math
import numpy as np
import json
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import queue
from kalmanFilter import kalman_filter, kalman_filter_cv, kalman_filter_cv2
import traceback
from plot_module import ClusterHandler

"""

Note if want to use self.TAS2, set self.TAS2 to true pick one of the self.TASs to save the dataset
meaning:
self.TAS = self.TAS1 | self.TAS3 | self.TAS4 | self.TAS5
self.TAS2 = True

"""


AXSIS_LMIT = 300 # Set the axis limit here
MAX_DISTANCE = 300 # Set the maximum distance here easily fliter out long distance points



# This class handles the LIDAR data processing and clustering
# It uses the YDLidarX2 class to get data and process it
# It also uses the DBSCAN algorithm for clustering
# The class is designed to be used in a multi-threaded environment

class lidarModules:
    def __init__(self, lidar, xaxsis=AXSIS_LMIT, yaxsis=AXSIS_LMIT, TASK = None, TASK2=False) -> None:
        self._lidar: YDLidarX2 = lidar
        self.scanStatus: bool = False
        self.GUI_PLOT = True # Too lazt to fix the GUI_PLOT variable, when flase chrashes dont touch it...
        self.debugInfo = True
        self.debug = False # hard coded debug

        if TASK == None:
            raise ValueError("TASK must be set to one of the tasks")
        if TASK not in ["Task1", "Task2", "Task3", "Task4", "Task5"]:
            raise ValueError("TASK must be set to one of the tasks")
        if TASK2 not in [True, False]:
            raise ValueError("TASK2 must be set to True or False")
        
        self.TASK = TASK 
        self.TASK2 = TASK2 

        self.limitedAngle: bool = True  # Limit the angle 
        if self.limitedAngle:
            self.angle0: float = 45
            self.angle1: float = 360 - 40
        else:
            self.angle0: float = 180
            self.angle1: float = 180
        self.MAX_DISTANCE: float = MAX_DISTANCE  # Maximum distance to consider for clustering

        self._dataset_records = []  # List to store dataset records
        # Settings for DBSCAN
        self.eps = 80  # It controls the local neighborhood of the points.
        self.min_samples = 5  # min_samples controls tolerance towards noise

        if self.GUI_PLOT:
            self.plotter = ClusterHandler(xaxsis=xaxsis, yaxsis=yaxsis)

        if self.TASK == "Task1": pass
        if self.TASK == "Task2": pass

        if self.TASK == "Task3": 
            self.kf = kalman_filter(lidar)
            print("Kalman filter 1 initialized")

        if self.TASK == "Task4": 
            self.kf = kalman_filter_cv(lidar)
            print("Kalman filter CV initialized")

        if self.TASK == "Task5": 
            self.kf = kalman_filter_cv2(lidar)
            print("Kalman filter CV2 initialized")

        self.missing_rate=0.2 # Probability of missing a measurement

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
            time.sleep(0.1)
            if self._lidar.available:
                distances = self._lidar.get_data()  # Use self._lidar consistently
                points = []
                for angle, dist in enumerate(distances):
                    if angle <= self.angle0 or angle >= self.angle1:
                        if dist <= 0 or dist > self.MAX_DISTANCE:
                            if self.debug: print(f"Distance out of range: {dist}")
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
                    match self.TASK:
                        case "Task1":
                            self.plotter.update_plot(points_np, labels)
                        case "Task2":
                            # Record dataset: select the cluster with the smallest Euclidean norm as the object.
                            pass
                        case "Task3" | "Task4":
                            self.plotter.update_plot(points_np, labels, kalman_state=self.kf.X)
                            if not cluster_means: continue
                            selected_label, selected_mean = min(cluster_means.items(), key=lambda item: np.linalg.norm(item[1])) 
                            if self.debugInfo: print(f"Selected Cluster {selected_label}: Mean position (mm): {selected_mean}")
                           
                            # Form the observation vector: use [x, y, 0, 0] (velocity unobserved)
                            z = np.array([selected_mean[0], selected_mean[1], 0, 0])
                            self.kf.predict()
                            self.kf.update(z)          
                            if self.debugInfo: print(f"Kalman filter state: {self.kf.X}")

                        case "Task5":
                            self.plotter.update_plot(points_np, labels, kalman_state=self.kf.X)
                            if not cluster_means: continue
                            selected_label, selected_mean = min(cluster_means.items(), key=lambda item: np.linalg.norm(item[1])) 
                            if self.debugInfo: print(f"Selected Cluster {selected_label}: Mean position (mm): {selected_mean}")
                            # Form observation vector: [x, y, 0, 0]
                            z = np.array([selected_mean[0], selected_mean[1], 0, 0])
                            
                            # Always predict, but sometimes skip the update to simulate missing observations.
                            self.kf.predict()
                            if np.random.rand() > self.missing_rate:
                                self.kf.update(z)
                                print(f"Update with measurement: {round(self.kf.X[0], 2)}, {round(self.kf.X[1], 2)}")
                            else:
                                print(f"Missing measurement, prediction only: {round(self.kf.X[0], 2)}, {round(self.kf.X[1], 2)}")
                            
                    for label, mean in cluster_means.items():
                        if self.debugInfo:
                            print(f"Cluster {label}: Mean position (mm): {mean}")   
                    if self.TASK2:
                        _dataset_recordsTemp = []
                        # Save the dataset: timestamp, x, y
                        timestamp = time.time()
                        for label, mean in cluster_means.items():
                            record = {
                                "timestamp": timestamp,
                                "x": mean[0],
                                "y": mean[1]
                            }
                            _dataset_recordsTemp.append(record)
                            print(f"Dataset record: {record}")
                        record = {
                            "One 360 >>>" : "#######################################",
                            "Dataset records": _dataset_recordsTemp
                        }
                        self._dataset_records.append(record)
                    print("#" * 20) 
                    if self.TASK2: self.save_dataset()

            time.sleep(0.1)
    def save_dataset(self, filename="dataset.json"):
        # Save the recorded dataset (timestamp, x, y) to a JSON file.

        #if filename is None:
        #    filename = f"dataset_{int(time.time())}.json"

        with open(filename, 'w') as f:
            json.dump(self._dataset_records, f, indent=4)
        print(f"Dataset saved to {filename}")
        

    def stopScan(self):self.scanStatus = False
    def _startScan(self):self.scanStatus = True

import matplotlib.pyplot as plt
import numpy as np
import math
import time
from ydlidar_x2 import YDLidarX2
from sklearn.cluster import DBSCAN


from lab3.YDLidarX2.kalmanFilter import *

def main():
    lidar = YDLidarX2('/dev/ttyUSB0')  # Adjust your COM port as needed
    print("Connection status", lidar.connect())
    lidar.start_scan()

    # Enable interactive plotting mode
    plt.ion()
    plt.figure()

    # Normal scan with graph for vis.
    #runScanCluster(lidar) # Task1
    #record_dataset(lidar, "dataset.json") Task2
    #runScanClusterKF(lidar)           # For clustering with Kalman filter integration (Task 3)
    #runScanClusterKF_CV(lidar)
    runScanClusterKF_CV_Missing(lidar)



    lidar.stop_scan()
    lidar.disconnect()
    print("Done")

if __name__ == "__main__":
    main()

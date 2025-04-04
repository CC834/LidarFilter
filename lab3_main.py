import matplotlib.pyplot as plt
import numpy as np
import math
import time
from ydlidar_x2 import YDLidarX2
from sklearn.cluster import DBSCAN
from lab3_modules import lidarModules

#from lab3.YDLidarX2.kalmanFilter import *

def main():
    lidar = YDLidarX2('/dev/ttyUSB0')  # Adjust your COM port as needed
    print("Connection status", lidar.connect())
    lidar.start_scan()

    # Create the lidar module with plotting/processing
    lidarMode = lidarModules(lidar)

    # Start scanning (this runs in a separate thread)
    lidarMode.runScan()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping scan...")

    # Stop scanning and disconnect
    lidarMode.stopScan()  # Stop our scanning thread
    lidar.stop_scan()
    lidar.disconnect()
    print("Done")

if __name__ == "__main__":
    main()

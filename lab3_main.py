import matplotlib.pyplot as plt
import numpy as np
import math
import time
from ydlidar_x2 import YDLidarX2
from sklearn.cluster import DBSCAN


from task1_module import *

def main():
    lidar = YDLidarX2('/dev/ttyUSB0')  # Adjust your COM port as needed
    print("Connection status", lidar.connect())
    lidar.start_scan()

    # Enable interactive plotting mode


    lidar.stop_scan()
    lidar.disconnect()
    print("Done")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import math
import time
from ydlidar_x2 import YDLidarX2
from sklearn.cluster import DBSCAN
from lab3_modules import lidarModules

#from lab3.YDLidarX2.kalmanFilter import *

def main():
    plt.ion()  
    plt.show(block=False)
    
    lidar = YDLidarX2('/dev/ttyUSB0')  # Adjust your COM port as needed
    print("Connection status", lidar.connect())
    lidar.start_scan()

    """

    Note if want to use self.TAS2, set self.TAS2 to true pick one of the self.TASs to save the dataset
    meaning:
    self.TAS = self.TAS1 | self.TAS3 | self.TAS4 | self.TAS5
    self.TAS2 = True

    """
    # Create the lidar module with plotting/processing
    lidarMode = lidarModules(lidar=lidar, xaxsis=300, yaxsis=300,TASK="Task5", TASK2=False)

    lidarMode.GUI_PLOT = True  # Set to true if you want to use the GUI plot
    lidarMode.debugInfo = True  # Set to true if you want to see the debug info

    print(lidarMode.TASK)

    # Start scanning (this runs in a separate thread)
    lidarMode.runScan()
    print("Scanning...")
    try:
        while True:
            lidarMode.plotter.update()  # Update the plot with new data
            #plt.pause(0.1)
    except KeyboardInterrupt:
        print("Stopping scan...")

    # Stop scanning and disconnect
    lidarMode.stopScan()  # Stop our scanning thread
    lidar.stop_scan()
    lidar.disconnect()
    print("Done")

if __name__ == "__main__":
    main()

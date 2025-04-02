from ydlidar_x2 import YDLidarX2
import time
import math

lidar = YDLidarX2('/dev/ttyUSB1')  # Your COM port
lidar.connect()
lidar.start_scan()

try:
    while True:
        if lidar.available:
            distances = lidar.get_data()  # Returns 360 distance values
            for angle, dist in enumerate(distances):
                #print(f"Angle: {angle}°, Distance: {dist} mm")
                radian = math.radians(angle)
                if angle >= 359:
                    exit()
                    
                x = dist * math.cos(radian)
                y = dist * math.sin(radian)
                print(f"X : {x} mm, Y: {y} mm >>>> angle : {angle} \n")
        # process the distance measurements
        time.sleep(0.1)

except KeyboardInterrupt:
    pass

lidar.stop_scan()
lidar.disconnect()
print("Done")
 
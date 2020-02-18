#!/usr/bin/env python
#delete something
# Yu Qi
import rospy
import utm
import serial
import math
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Int64
from lab2.msg import yu

def talker():
    pub = rospy.Publisher('gps_chat', yu, queue_size=10)
    rospy.init_node('gps_xy', anonymous=True)
    
    portgps = "$GPGGA,134658.00,5106.9792,N,11402.3003,W,2,09,1.0,1048.47,M,-16.27,M,08,AAAA*60"
    # portgps = serial.Serial('/dev/xxxxxxxxx', baudrate=4800)
    # portimu = serial.Serial('/dev/xxxxxxxxx', baudrate=115200)
    portimu = "VNYMR,-045.286,-003.093,+000.408,+00.2007,+00.1929,+00.3346,-00.498,-00.047,-09.705,-00.00143,+00.012139,-00.000167*67"

    while not rospy.is_shutdown():
        
        # datagps = portgps.readline()
        # dataimu = portimu.readline()

        datagps = portgps
        dataimu = portimu

        if(datagps == "" or dataimu == ""):
            rospy.logwarn("no data")
        else:
            if(datagps.startswith('$GPGGA') and dataimu.startswith('VNYMR')):
                datasplitgps = datagps.split(",")
                datasplitimu = dataimu.split(",")
                ##############gps##############
                latitute = datasplitgps[2]
                latitute_deg = float(latitute[0:2])
                latitute_min = float(latitute[2:])
                latitute = float(latitute)
                latitute_utm = latitute_deg+latitute_min
                longitite = datasplitgps[4]
                longitite_deg = float(longitite[0:3])
                longitite_min = float(longitite[3:])
                longitite = float(longitite)
                longitite_utm = longitite_deg+longitite_min
                altt = float(datasplitgps[9])
                # REF : https://github.com/Turbo87/utm
                utm_res = utm.from_latlon(latitute_utm, longitite_utm)
                ##############gps##############
                ##############imu##############
                yaw = float(datasplitimu[1])
                pitch = float(datasplitimu[2])
                roll = float(datasplitimu[3])
                magnetic_x = float(datasplitimu[4])
                magnetic_y = float(datasplitimu[5])
                magnetic_z = float(datasplitimu[6])
                acc_x = float(datasplitimu[7])
                acc_y = float(datasplitimu[8])
                acc_z = float(datasplitimu[9])
                angular_x = float(datasplitimu[10])
                angular_y = float(datasplitimu[11])
                l = len(datasplitimu[12])
                temp = datasplitimu[12]
                angular_z = float(temp[:l-3])

                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                cp = math.cos(pitch * 0.5)
                sp = math.sin(pitch * 0.5)
                cr = math.cos(roll * 0.5)
                sr = math.sin(roll * 0.5)

                w = cy * cp * cr + sy * sp * sr
                x = cy * cp * sr - sy * sp * cr
                y = sy * cp * sr + cy * sp * cr
                z = sy * cp * cr - cy * sp * sr
                ##############imu##############
                msg = yu()
                msg.header = "gpgga"
                msg.latitude = latitute
                msg.longitude = longitite
                msg.alt = altt
                msg.utm_easting = utm_res[0]
                msg.utm_northing = utm_res[1]
                msg.zone = utm_res[2]
                msg.letter = utm_res[3]

                imu_x = x
                imu_y = y
                imu_z = z
                imu_w = w
                msg.magnetic_x = magnetic_x
                msg.magnetic_y = magnetic_y
                msg.magnetic_z = magnetic_z
                msg.acc_x = acc_x
                msg.acc_y = acc_y
                msg.acc_z = acc_z
                msg.angular_x = angular_x
                msg.angular_y = angular_y
                msg.angular_z = angular_z

                rospy.loginfo(msg)
                pub.publish(msg)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python
#delete something
# Yu Qi
import rospy
import utm
import serial
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Int64
from lab1.msg import yu

def talker():
    pub = rospy.Publisher('gps_chat', yu, queue_size=10)
    rospy.init_node('gps_xy', anonymous=True)
    # rate = rospy.Rate(10) # 10hz

    # because I do not have gps, code this part by imagination
    # but I write a string of gpgga format to data below
    # port = serial.Serial('/dev/xxxxxxxxx', baudrate=4800)

    while not rospy.is_shutdown():
        
        # data = port.readline()
        data = "$GPGGA,134658.00,5106.9792,N,11402.3003,W,2,09,1.0,1048.47,M,-16.27,M,08,AAAA*60"
        # REF : https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm

        if(data ==""):
            rospy.logwarn("no data")
        else:
            if(data.startswith('$GPGGA')):
                datasplit = data.split(",")

                latitute = datasplit[2]
                latitute_deg = float(latitute[0:2])
                latitute_min = float(latitute[2:])
                latitute = float(latitute)
                latitute_utm = latitute_deg+latitute_min


                longitite = datasplit[4]
                longitite_deg = float(longitite[0:3])
                longitite_min = float(longitite[3:])
                longitite = float(longitite)
                longitite_utm = longitite_deg+longitite_min


                altt = float(datasplit[9])

                # REF : https://github.com/Turbo87/utm
                utm_res = utm.from_latlon(latitute_utm, longitite_utm)

                msg = yu()
                msg.header = "gpgga"
                msg.latitude = latitute
                msg.longitude = longitite
                msg.alt = altt
                msg.utm_easting = utm_res[0]
                msg.utm_northing = utm_res[1]
                msg.zone = utm_res[2]
                msg.letter = utm_res[3]

                # string header
                # float64 latitude
                # float64 longitude
                # float64 alt
                # float64 utm_easting
                # float64 utm_northing
                # Int64 zone
                # string letter

                rospy.loginfo(msg)
                pub.publish(msg)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

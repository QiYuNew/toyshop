import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import utm
import codecs
import string
from scipy import signal
from mpl_toolkits import mplot3d
from math import radians, cos, sin, asin, sqrt
import turtle

PI = 3.1415926
filepath1 = './gps.csv'
filepath2 = './imu.csv'
filepath3 = './mag.csv'

gpsf = len(open(filepath1).readlines())
imuf = len(open(filepath2).readlines())
magf = len(open(filepath3).readlines())

la_gps = np.zeros([gpsf,1])
al_gps = np.zeros([gpsf,1])

imu_time = np.zeros([imuf,1])
qx = np.zeros([imuf,1])
qy = np.zeros([imuf,1])
qz = np.zeros([imuf,1])
qw = np.zeros([imuf,1])
imu_avx = np.zeros([imuf,1])
imu_avy = np.zeros([imuf,1])
imu_avz = np.zeros([imuf,1])
imu_accx = np.zeros([imuf,1])
imu_accy = np.zeros([imuf,1])
imu_accz = np.zeros([imuf,1])

mag_x = np.zeros([magf,1])
mag_y = np.zeros([magf,1])
mag_z = np.zeros([magf,1])

roll = np.zeros([imuf,1])
pitch = np.zeros([imuf,1])
yaw = np.zeros([imuf,1])

f = codecs.open('./imu.txt', mode='r', encoding='utf-8')
line = f.readline()
list1 = []
i = 0
while line:
    a = line.split(',')
    b = a[2:3]
    imu_time[i, 0] = float(str(b)[7:len(b)-3])
    i += 1
    line = f.readline()
f.close()

i = 0
with open(filepath1, 'r') as file:
    reader = csv.reader(file)
    for data in reader:
        la_gps[i, 0] = float(data[4])
        al_gps[i, 0] = float(data[5])
        i += 1

i = 0
with open(filepath2, 'r') as file:
    reader = csv.reader(file)
    for data in reader:
        qx[i, 0] = float(data[4])
        qy[i, 0] = float(data[5])
        qz[i, 0] = float(data[6])
        qw[i, 0] = float(data[7])
        imu_avx[i, 0] = float(data[17])
        imu_avy[i, 0] = float(data[18])
        imu_avz[i, 0] = float(data[19])
        imu_accx[i, 0] = float(data[29])
        imu_accy[i, 0] = float(data[30])
        imu_accz[i, 0] = float(data[31])
        i += 1

i = 0
with open(filepath3, 'r') as file:
    reader = csv.reader(file)
    for data in reader:
        mag_x[i, 0] = float(data[4])
        mag_y[i, 0] = float(data[5])
        mag_z[i, 0] = float(data[6])
        i += 1

# lowpass mag
b_mag, a_mag = signal.butter(5, 0.02, 'lowpass')
temp_mag_x = np.zeros([1, imuf])
temp_mag_y = np.zeros([1, imuf])
temp_mag_z = np.zeros([1, imuf])
temp_mag_x = signal.filtfilt(b_mag, a_mag, mag_x.T)
temp_mag_y = signal.filtfilt(b_mag, a_mag, mag_y.T)
temp_mag_z = signal.filtfilt(b_mag, a_mag, mag_z.T)
mag_x = temp_mag_x.T
mag_y = temp_mag_y.T
mag_z = temp_mag_z.T

# lowpass acc
# b_acc, a_acc = signal.butter(5, 0.005, 'lowpass')
# temp_acc_x = np.zeros([1, imuf])
# temp_acc_y = np.zeros([1, imuf])
# temp_acc_z = np.zeros([1, imuf])
# temp_acc_x = signal.filtfilt(b_acc, a_acc, imu_accx.T)
# temp_acc_y = signal.filtfilt(b_acc, a_acc, imu_accy.T)
# temp_acc_z = signal.filtfilt(b_acc, a_acc, imu_accz.T)
# imu_accx = temp_acc_x .T
# imu_accy = temp_acc_y.T
# imu_accz = temp_acc_z.T

# q to ypr
for i in range(len(qx)):
    #  roll
    sinr_cosp = 2 * (qw[i, 0] * qx[i, 0] + qy[i, 0] * qz[i, 0])
    cosr_cosp = 1 - 2 * (qx[i, 0] * qx[i, 0] + qy[i, 0] * qy[i, 0])
    roll[i, 0] = math.atan2(sinr_cosp, cosr_cosp)
    #pitch
    sinp = 2 * (qw[i, 0] * qy[i, 0] - qz[i, 0] * qx[i, 0])
    if(abs(sinp) >= 1):
        pitch[i, 0] = math.copysign(PI/2, sinp)
    else:
        pitch[i, 0] = math.asin(sinp)
    #yaw
    siny_cosp = 2 * (qw[i, 0] * qz[i, 0] + qx[i, 0] * qy[i, 0])
    cosy_cosp = 1 - 2 * (qy[i, 0]*qy[i, 0] + qz[i, 0] * qz[i, 0])
    yaw[i, 0] = math.atan2(siny_cosp, cosy_cosp)

# gps
plt.figure(1)
plt.title("gps")
plt.plot(la_gps, al_gps, '+')

yaw_coor = np.zeros([imuf,1])
yaw_coor = yaw
# ypr
plt.figure(2)
plt.title("pry")
plt.plot(imu_time, pitch, linewidth=1, label = 'pitch')
plt.plot(imu_time, roll, linewidth=1, label = 'roll')
plt.plot(imu_time, yaw, linewidth=1, label = 'yaw')
plt.legend()

# acc
plt.figure(3)#57470
plt.title("acc")
plt.plot(imu_time, imu_accy, linewidth=1, label = 'y')
plt.plot(imu_time, imu_accz, linewidth=1, label = 'z')
plt.plot(imu_time, imu_accx, linewidth=1, label = 'x')
plt.legend()

# mag
# plt.figure(4)
# plt.title("mag")
# plt.plot(imu_time, mag_x, linewidth=1, label = 'x')
# plt.plot(imu_time, mag_y, linewidth=1, label = 'y')
# plt.plot(imu_time, mag_z, linewidth=1, label = 'z')
# plt.legend()

# mag calibration - circle
mag_ex_x = np.zeros([5000,1])
mag_ex_y = np.zeros([5000,1])
mag_ex_z = np.zeros([5000,1])
mag_ex_xyz = np.zeros([5000,3])

imu_time_ex = imu_time[4000:9000, 0]
mag_ex_x = mag_x[4000:9000, 0]
mag_ex_y = mag_y[4000:9000, 0]
mag_ex_z = mag_z[4000:9000, 0]
# mag_ex_xyz[:, 0] = mag_x[4000:9000, 0]
# mag_ex_xyz[:, 1] = mag_y[4000:9000, 0]
# mag_ex_xyz[:, 2] = mag_z[4000:9000, 0]

# plt.figure(99)
# ax = plt.axes(projection='3d')
# # ax.set_ylim(-0.05, 0.05)
# # ax.set_xlim(-0.05, 0.05)
# # ax.set_zlim(-0.05, 0.05)
# ax.scatter3D(mag_ex_x, mag_ex_y, mag_ex_z, marker = '.')

mag_x_bais = abs((max(mag_ex_x)+min(mag_ex_x))/2)
mag_y_bais = abs((max(mag_ex_y)+min(mag_ex_y))/2)
mag_z_bais = abs((max(mag_ex_z)+min(mag_ex_z))/2)

mag_x_scale = (max(mag_ex_x)-min(mag_ex_x))/2
mag_y_scale = (max(mag_ex_y)-min(mag_ex_y))/2
mag_z_scale = (max(mag_ex_z)-min(mag_ex_z))/2
avg_rad = (mag_x_scale+mag_y_scale+mag_z_scale)/3
x_scale = avg_rad/mag_x_scale
y_scale = avg_rad/mag_y_scale
z_scale = avg_rad/mag_z_scale

# print(mag_x_bais)
# print(mag_y_bais)
# print(mag_z_bais)
# print(x_scale)
# print(y_scale)
# print(z_scale)
# 0.029500625627526242
# 0.2178967636798682
# 0.07160614838188292
# 0.7645566274387674
# 0.7411241238820084
# 2.917571922941182
# for i in range(5000):
#     mag_ex_x[i] = (mag_ex_x[i] - mag_x_bais)*x_scale
#     mag_ex_y[i] = (mag_ex_y[i] + mag_y_bais)*y_scale
#     mag_ex_z[i] = (mag_ex_z[i] + mag_z_bais)*z_scale

# plt.figure(98)
# plt.title("mag - after calibration")
# plt.plot(imu_time_ex, mag_ex_x, linewidth=1, label = 'x')
# plt.plot(imu_time_ex, mag_ex_y, linewidth=1, label = 'y')
# plt.plot(imu_time_ex, mag_ex_z, linewidth=1, label = 'z')
# plt.legend()

# plt.figure(97)
# ax = plt.axes(projection='3d')
# plt.title("mag - after calibration")
# ax.set_ylim(-0.05, 0.05)
# ax.set_xlim(-0.05, 0.05)
# ax.set_zlim(-0.05, 0.05)
# ax.scatter3D(mag_ex_x, mag_ex_y, mag_ex_z, marker = '.')


# 0.029500625627526242
# 0.2178967636798682
# 0.07160614838188292
# 0.7645566274387674
# 0.7411241238820084
# 2.917571922941182
mag_x_bais = 0.029500625627526242
mag_y_bais = 0.2178967636798682
mag_z_bais = 0.07160614838188292
x_scale = 0.7645566274387674
y_scale = 0.7411241238820084
z_scale = 2.917571922941182

# cal all
for i in range(len(mag_x)):
    mag_x[i] = (mag_x[i] - mag_x_bais)*x_scale
    mag_y[i] = (mag_y[i] + mag_y_bais)*y_scale
    mag_z[i] = (mag_z[i] + mag_z_bais)*z_scale

plt.figure(5)
plt.title("mag")
plt.plot(imu_time, mag_x, linewidth=1, label = 'x')
plt.plot(imu_time, mag_y, linewidth=1, label = 'y')
plt.plot(imu_time, mag_z, linewidth=1, label = 'z')
plt.legend()




# plt.show()
######################2222222222222222222222222#############################


################calcaute v from gps
t_gps = 1
def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

dis_gps = np.zeros([len(la_gps), 1])
v_gps = np.zeros([len(la_gps), 1])
for i in range(1, len(la_gps)):
    dis = 0
    dis = geodistance(la_gps[i-1], al_gps[i-1], la_gps[i], al_gps[i])
    dis_gps[i-1] = 1000*dis
    v_gps[i-1] = 1000*dis/t_gps
################calcaute v from gps


################calcaute v from imu - acc
t_imu = 0.025
v_int = np.zeros([len(imu_accx), 1])
for i in range(1, len(imu_accx)):
    temp = (imu_accx[i]+0.22)*t_imu + v_int[i-1]
    if(temp >= 0):
        v_int[i] = temp
    else:
        v_int[i] = 0

################calcaute v from imu - acc


#57470
#1438
gps_plot_x = np.arange(0,1438)

plt.figure(6)
plt.title("acc")
plt.plot(imu_time, v_int, linewidth=1, label = 'int x acc to V')

plt.legend()

plt.figure(7)
plt.plot(gps_plot_x, v_gps, linewidth=1, label = 'int x acc to V')

imu_avz_int =  np.zeros([len(imu_avz), 1])
imu_avz_mean = np.mean(imu_avz)
# print(imu_avz_mean)
for i in range(len(imu_avz)):
    imu_avz[i] -= imu_avz_mean

for i in range(len(imu_avz)):
    imu_avz_int[i] = imu_avz[i]*t_imu

plt.figure(8)
plt.plot(imu_time, imu_avz_int, linewidth=1, label = 'yaw rate int')


###################3333333333333333333333333333
t_navi = 0.025
dis_acc_int = np.zeros([len(v_int), 1])
for i in range(len(v_int)):
    if(v_int[i] > 0):
        dis_acc_int[i] = v_int[i]*t_navi
    else:
        dis_acc_int[i] = 0
    


yaw_navi = np.zeros([45000-9000+1, 1])
dis_navi = np.zeros([45000-9000+1, 1])
yaw_navi = yaw_coor[9000:45000,0]
dis_navi = dis_acc_int[9000:45000,0]

coor = np.zeros([len(dis_navi), 2])# x | y
coor[0,:] = [0, 0]
yaw_turn = 0
yaw_accumu = 0
cnt1 = []
cnt2 = []

for i in range(1, len(dis_navi)):
    yaw_turn = (yaw_navi[i]-yaw_navi[i-1])/PI*180
    if(-2.4 < yaw_navi[i] < -1.5 and i < 21000):
        coor[i,0] = coor[i-1,0] + abs(dis_navi[i])*sin(abs(yaw_turn))# x
        coor[i,1] = coor[i-1,1] + abs(dis_navi[i])*cos(abs(yaw_turn))# y
    elif(-1.5 < yaw_navi[i] < -0.5 and i < 21000):
        coor[i,0] = coor[i-1,0] + abs(dis_navi[i])*cos(abs(yaw_turn))# x
        coor[i,1] = coor[i-1,1] + abs(dis_navi[i])*sin(abs(yaw_turn))# y
        temp_x = coor[i,0]
        temp_y = coor[i,1]
    elif(0 < yaw_navi[i] < 1 and i < 21000 and i == 10339):
        coor[10339,1] = temp_y
        coor[10339,0] = temp_x
    elif(0 < yaw_navi[i] < 1 and i < 21000 and i > 10339):
        coor[i,1] = coor[i-1,1] - abs(dis_navi[i])*cos(abs(yaw_turn))# y
        coor[i,0] = coor[i-1,0] - abs(dis_navi[i])*sin(abs(yaw_turn))# x
        temp_x = coor[i,0]
        temp_y = coor[i,1]
    elif(1.5 < yaw_navi[i] < 2.5 and i < 21000 and i == 18909):
        coor[18909,1] = temp_y
        coor[18909,0] = temp_x
    elif(1.5 < yaw_navi[i] < 2.5 and i < 21000 and i > 18909):
        coor[i,0] = coor[i-1,0] - abs(dis_navi[i])*cos(abs(yaw_turn))# x
        coor[i,1] = coor[i-1,1] + abs(dis_navi[i])*sin(abs(yaw_turn))# y
    elif(1.5 < yaw_navi[i] < 2.5 and i > 21000 and i > 18909):
        coor[i,0] = coor[i-1,0] - abs(dis_navi[i])*cos(abs(yaw_turn))# x
        coor[i,1] = coor[i-1,1] + abs(dis_navi[i])*sin(abs(yaw_turn))# y
        cnt1.append(i)
        temp_x = coor[i,0]
        temp_y = coor[i,1]
    elif(0 < yaw_navi[i] < 1 and i > 21000 and i == 23693):
        coor[23693,1] = temp_y
        coor[23693,0] = temp_x
        cnt2.append(i)
    elif(0 < yaw_navi[i] < 1  and i > 21000 and i > 23693):
        coor[i,0] = coor[i-1,0] - abs(dis_navi[i])*sin(abs(yaw_turn))# x
        coor[i,1] = coor[i-1,1] - abs(dis_navi[i])*cos(abs(yaw_turn))# y
# print(coor[23630,1])
        


    

    # yaw_turn = (yaw_navi[i]-yaw_navi[i-1])/PI*180
    # yaw_accumu += yaw_turn
    # if(abs(yaw_accumu) < 90):
    #     flag = 0
    # elif(90 < abs(yaw_accumu) < 180):
    #     flag = 1

    # if(flag == 0):
    #     if(yaw_turn >= 0):#right
    #         coor[i,0] = coor[i-1,0] + abs(dis_navi[i])*sin(abs(yaw_turn))# x
    #         coor[i,1] = coor[i-1,1] + abs(dis_navi[i])*cos(abs(yaw_turn))# y     
    #     elif(yaw_turn < 0):
    #         coor[i,0] = coor[i-1,0] - abs(dis_navi[i])*sin(abs(yaw_turn))# x
    #         coor[i,1] = coor[i-1,1] + abs(dis_navi[i])*cos(abs(yaw_turn))# y
    # elif(flag == 1):
    #     if(yaw_turn >= 0):#right
    #         coor[i,0] = coor[i-1,0] + abs(dis_navi[i])*cos(abs(yaw_turn))# x
    #         coor[i,1] = coor[i-1,1] - abs(dis_navi[i])*sin(abs(yaw_turn))# y     
    #     elif(yaw_turn < 0):
    #         coor[i,0] = coor[i-1,0] + abs(dis_navi[i])*cos(abs(yaw_turn))# x
    #         coor[i,1] = coor[i-1,1] + abs(dis_navi[i])*sin(abs(yaw_turn))# y
    # if(-270<yaw_accumu<-180 or 180<yaw_accumu<270):
    #     if(yaw_turn >= 0):#right
    #         coor[i,0] = coor[i-1,0] - dis_navi[i]*sin(abs(yaw_turn))# x
    #         coor[i,1] = coor[i-1,1] - dis_navi[i]*cos(abs(yaw_turn))# y     
    #     elif(yaw_turn < 0):
    #         coor[i,0] = coor[i-1,0] + dis_navi[i]*sin(abs(yaw_turn))# x
    #         coor[i,1] = coor[i-1,1] - dis_navi[i]*cos(abs(yaw_turn))# y
    # else:
    #     continue
      

    
print(cnt1[-1], cnt2[0], cnt2[-1])
plt.figure(101)
plt.plot(coor[:21000,0], coor[:21000,1], '+')

plt.figure(102)
i = 45000
plt.plot(imu_time[9000:i,0], yaw_navi[:i-9000], linewidth=5)

plt.show()

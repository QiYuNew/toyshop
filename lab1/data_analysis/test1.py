import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import utm

filepath1 = './location.csv'
filepath2 = './location2.csv'
Lstation = len(open(filepath1).readlines())
Lmoving = len(open(filepath2).readlines())

la_station = np.zeros([Lstation,1])
al_station = np.zeros([Lstation,1])
la_moving = np.zeros([Lmoving,1])
al_moving = np.zeros([Lmoving,1])

i = 0
with open(filepath1, 'r') as file:
    reader = csv.reader(file)
    for data in reader:
        la_station[i, 0] = data[1]
        al_station[i, 0] = data[2]
        i += 1

i = 0
with open(filepath2, 'r') as file:
    reader = csv.reader(file)
    for data in reader:
        la_moving[i, 0] = data[1]
        al_moving[i, 0] = data[2]
        i += 1


plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(la_station, al_station)
plt.subplot(1, 2, 2)
plt.plot(la_moving, al_moving)
plt.show()

import CF_functions as cff
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
import numpy.ma as ma
import math

Kpos_P = np.array([10,10,10])
Kpos_P_limit = 100
Kpos_D = np.array([7,7,7])
Kpos_D_limit = 100
Kpos_I = np.array([3,3,3])
Kpos_I_limit = 2.0
mass = 0.033

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="logfile")
args = parser.parse_args()

# decode binary log data
logData = cff.decode(args.file)

# set window background to white
# plt.rcParams['figure.facecolor'] = 'w'


statePos = np.column_stack((
  logData['stateEstimateZ.x'] / 1000.0,
  logData['stateEstimateZ.y'] / 1000.0,
  logData['stateEstimateZ.z'] / 1000.0))

stateVel = np.column_stack((
  logData['stateEstimateZ.vx'] / 1000.0,
  logData['stateEstimateZ.vy'] / 1000.0,
  logData['stateEstimateZ.vz'] / 1000.0))

pos_d = np.column_stack((
  logData['ctrltargetZ.x'] / 1000.0,
  logData['ctrltargetZ.y'] / 1000.0,
  logData['ctrltargetZ.z'] / 1000.0))

vel_d = np.column_stack((
  logData['ctrltargetZ.vx'] / 1000.0,
  logData['ctrltargetZ.vy'] / 1000.0,
  logData['ctrltargetZ.vz'] / 1000.0))

i_error_pos = np.column_stack((
  logData['ctrlLee.i_error_posx'],
  logData['ctrlLee.i_error_posy'],
  logData['ctrlLee.i_error_posz']))


pos_e = np.clip(pos_d - statePos, -Kpos_P_limit, Kpos_P_limit)
vel_e = np.clip(vel_d - stateVel, -Kpos_D_limit, Kpos_D_limit)

Ppart = mass * Kpos_P * pos_e
Dpart = mass * Kpos_D * vel_e
Ipart = mass * Kpos_I * i_error_pos
FWpart = np.repeat([[0,0,mass * 9.81]], Ppart.shape[0],axis=0)

F_d = Ppart + Dpart + Ipart + FWpart
thrust = np.linalg.norm(F_d, axis=1)

yaw = 0
roll = np.arcsin((F_d[:,0] * np.sin(yaw) - F_d[:,1] * np.cos(yaw)) / thrust)
pitch = np.arctan(F_d[:,0] * np.cos(yaw) + F_d[:,1] * np.sin(yaw)) / F_d[:,2]

# number of columns and rows for suplot
plotCols = 3
plotRows = 2

# current plot for simple subplot usage
plotCurrent = 0

for i, axis in enumerate(['x', 'y', 'z']):

  plotCurrent = i + 1
  plt.subplot(plotRows, plotCols, plotCurrent)
  plt.stackplot(logData['tick'] / 1e6, 
    Ppart[:,i] / 9.81 * 1000,
    Dpart[:,i] / 9.81 * 1000,
    Ipart[:,i] / 9.81 * 1000,
    FWpart[:,i] / 9.81 * 1000,
    labels=["P","D", "I", "FW"],alpha=0.6)
  plt.xlabel('Time [s]')
  plt.ylabel('Thrust [g]'.format(axis))
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

plotCurrent = 4
plt.subplot(plotRows, plotCols, plotCurrent)
plt.plot(logData['tick'] / 1e6, thrust / 9.81 * 1000)
plt.xlabel('Time [s]')
plt.ylabel('Desired motor force [g]')
plt.legend(loc=9, ncol=3, borderaxespad=0.)

plotCurrent = 5
plt.subplot(plotRows, plotCols, plotCurrent)
plt.plot(logData['tick'] / 1e6, np.degrees(roll), label='roll')
plt.plot(logData['tick'] / 1e6, np.degrees(pitch), label='pitch')
plt.xlabel('Time [s]')
plt.ylabel('Desired attitude [deg]')
plt.legend(loc=9, ncol=3, borderaxespad=0.)

# # print(FWpart)
# # print(mass * 9.81 / 9.81 * 1000)
# Ppart = mass * (Kpos_P * pos_e)[:,0] / 9.81 * 1000
# Dpart = mass * (Kpos_D * vel_e)[:,0] / 9.81 * 1000
# Ipart = mass * (Kpos_I * i_error_pos)[:,0] / 9.81 * 1000

# FWpart = np.repeat(mass * 9.81 / 9.81 * 1000, Ppart.shape[0])
# FWpart = np.repeat(0, Ppart.shape[0])

# plt.stackplot(logData['tick'] / 1e6, 
# 	Ppart,
# 	Dpart,
# 	Ipart,
# 	FWpart,
# 	labels=["P","D", "I", "FW"])
# # plt.plot(logData['tick'] / 1e6, pos_e)
# plt.legend()
plt.show()

# plt.xlabel('Time [s]')
# plt.ylabel('Thrust mixing [g]'.format(axis))
# plt.legend(loc=9, ncol=3, borderaxespad=0.)


import CF_functions as cff
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
import numpy.ma as ma
import math

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="logfile")
args = parser.parse_args()

# decode binary log data
logData = cff.decode(args.file)

# set window background to white
plt.rcParams['figure.facecolor'] = 'w'
    
# number of columns and rows for suplot
plotCols = 3
plotRows = 4

# current plot for simple subplot usage
plotCurrent = 0

time = logData['tick'] / 1e6

for i, axis in enumerate(['x', 'y', 'z']):

	plotCurrent = i + 1
	plt.subplot(plotRows, plotCols, plotCurrent)
	plt.plot(time, np.degrees(logData['ctrlLee.rpy' + axis]), '-', label='state')
	plt.plot(time, np.degrees(logData['ctrlLee.rpyd' + axis]), '-', label='target')
	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Angle [deg]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

	plotCurrent = i + 4
	plt.subplot(plotRows, plotCols, plotCurrent)
	plt.plot(time, np.degrees(logData['ctrlLee.omega' + axis]), '-', label='w')
	plt.plot(time, np.degrees(logData['ctrlLee.omegar' + axis]), '-', label='w_r')
	# plt.plot(time, np.degrees(logData['ctrlLee.omega' + axis] - logData['ctrlLee.omegar' + axis]), '-', label='w_e')
	# plt.plot(time, np.degrees(logData['ctrlLee.i_error_att' + axis]), '-', label='accumulated error')

	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Angular velocity [deg/s]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

	plotCurrent = i + 7
	plt.subplot(plotRows, plotCols, plotCurrent)
	plt.plot(time, logData['stateEstimateZ.' + axis] / 1000.0, '-', label='state')
	plt.plot(time, logData['ctrltargetZ.' + axis] / 1000.0, '-', label='target')
	# plt.plot(time, logData['stateEstimateZ.' + axis] / 1000.0 - logData['ctrltargetZ.' + axis] / 1000.0, '-', label='error')
	plt.plot(time, logData['ctrlLee.i_error_pos' + axis], '-', label='accumulated error')
	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Position [m]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

	plotCurrent = i + 10
	plt.subplot(plotRows, plotCols, plotCurrent)
	plt.plot(time, logData['stateEstimateZ.v' + axis] / 1000.0, '-', label='state')
	plt.plot(time, logData['ctrltargetZ.v' + axis] / 1000.0, '-', label='target')
	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Velocity [m/s]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

	# plotCurrent = i + 10
	# plt.subplot(plotRows, plotCols, plotCurrent)
	# plt.plot(time, logData['pwm.m1_pwm'] / 65536.0, '-', label='m1')
	# plt.plot(time, logData['pwm.m2_pwm'] / 65536.0, '-', label='m2')
	# plt.plot(time, logData['pwm.m3_pwm'] / 65536.0, '-', label='m3')
	# plt.plot(time, logData['pwm.m4_pwm'] / 65536.0, '-', label='m4')
	# # plt.plot(time, logData['motor.saturation'], '-', label='saturation')
	# # plt.plot(time, logData['ctrlLee.torquex'], '-', label='torquex')
	# # plt.plot(time, logData['ctrlLee.torquey'], '-', label='torquey')
	# # plt.plot(time, logData['ctrlLee.torquez'], '-', label='torquez')
	# plt.xlabel('RTOS Ticks [ms]')
	# plt.ylabel('Normalized motor output')
	# plt.legend(loc=9, ncol=3, borderaxespad=0.)


plotCurrent = 10
plt.subplot(plotRows, plotCols, plotCurrent)
plt.plot(time, logData['motor.f1'] / 9.81 * 1000, '-', label='f1')
plt.plot(time, logData['motor.f2'] / 9.81 * 1000, '-', label='f2')
plt.plot(time, logData['motor.f3'] / 9.81 * 1000, '-', label='f3')
plt.plot(time, logData['motor.f4'] / 9.81 * 1000, '-', label='f4')
plt.plot(time, logData['pwm.maxThrust'], '--')

plt.xlabel('Time [s]')
plt.ylabel('Desired motor force [g]'.format(axis))
plt.legend(loc=9, ncol=3, borderaxespad=0.)


plt.show()

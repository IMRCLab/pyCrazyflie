#!/usr/bin/env python
import sys
import os
path = os.path
sys.path.append('uavDy/')
sys.path.append('Utilities/')
sys.path.append('simulator/')
sys.path.append('genTrajectory/')
sys.path.append('trajectoriescsv/')
sys.path.append('crazyflie-firmware/')

import cffirmware
import uav
import numpy as np
from initialize import dt, initState
from rowan import to_euler, from_matrix, to_matrix , from_euler
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import matplotlib.animation as animation
from initialize import initState, dt
from AnimateSingleUav import PlotandAnimate
import time

def main():
    # Initialize an instance of a uav dynamic model with:
    # dt: time interval
    # initState: initial state
    # set it as 1 tick: i.e: 1 ms
    uav1 = uav.UavModel(dt, initState)
    # Upload the traj in csv file format
    # rows: time, xdes, ydes, zdes
    filename = "trajectoriescsv/lin.csv"
    timeStamped_traj = np.genfromtxt(filename, delimiter=',')
    # final time of traj in ms
    tf_ms = timeStamped_traj[0,-1]*1e3
    
    #initialize the controller and allocate current state (both sensor and state are the state)
    # This is kind of odd and should be part of state
    control, setpoint, sensors, state = initController()
    print("initial values: \n")
    print("initial thrust: %d\n",control.thrustSI)
    print("initial torque: ",control.torque[0],control.torque[1], control.torque[2])
    print("initial position", state.position.x,state.position.y,state.position.z)
    
    # Note that 1 tick == 1ms
    # note that the attitude controller will only compute a new output at 500 Hz
    # and the position controller only at 100 Hz
    # If you want an output always, simply select tick==0
    full_state = np.zeros((1,13))
    
    for tick in range(0, int(tf_ms)+1):
        # update desired state
        setpoint  = updateDesState(tick, setpoint, timeStamped_traj[1::,tick])
        # update current state
        state     = updateState(state, uav1.state)
        sensors   = updateSensor(sensors, uav1.state)
        # query the controller
        cffirmware.controllerSJC(control, setpoint, sensors, state, tick)
        control_inp =  np.array([control.thrustSI, control.torque[0], control.torque[1], control.torque[2]])
        uav1.states_evolution(control_inp)

        print("thrust: %d\n",control.thrustSI)
        print("torque: ",control.torque[0],control.torque[1], control.torque[2])
        print("position", state.position.x,state.position.y,state.position.z)
        print("rpy", state.attitude.roll,state.attitude.pitch,state.attitude.yaw)
    
        full_state = np.concatenate((full_state, uav1.state.reshape(1,13)))
        

    full_state = np.delete(full_state, 0, 0)
    fig     = plt.figure(figsize=(10,10))
    ax      = fig.add_subplot(autoscale_on=True,projection="3d")
    sample  = 50
    animate = PlotandAnimate(fig, ax, uav1, full_state[::sample,:])

    animateAndSave = True
    if animateAndSave:
        videoname  = 'hover.mp4' 
        dt_sampled = dt * sample
        show       = False
        save       = True
        if show:
            print("Showing animation.")
        if save:
            print("Converting Animation to Video. \nPlease wait...")
        now = time.time()
        startanimation = animate.startAnimation(videoname,show,save,dt_sampled)
        # plotfulltraj = animate.plotFulltraj();
        end = time.time()
        print("Run time:  {:.3f}s".format((end - now)))
    else:
        print("plotting full trajectory")
        plotfulltraj = animate.plotFulltraj();
##----------------------------------------------------------------------------------------------------------------------------------------------------------------##        
##----------------------------------------------------------------------------------------------------------------------------------------------------------------##
def initController():
    cffirmware.controllerSJCInit()
    # Allocate output variable
    # For this example, only thrustSI, and torque members are relevant
    control = cffirmware.control_t()
    # allocate desired state
    setpoint = cffirmware.setpoint_t()
    setpoint = setTrajmode(setpoint)
    sensors = cffirmware.sensorData_t()
    state = cffirmware.state_t()
    return control, setpoint, sensors, state 

def setTrajmode(setpoint):
    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.mode.roll = cffirmware.modeDisable
    setpoint.mode.pitch = cffirmware.modeDisable
    setpoint.mode.yaw = cffirmware.modeDisable
    return setpoint

def updateDesState(tick, setpoint, fulltraj):
    setpoint.position.x = fulltraj[0]  # m
    setpoint.position.y = fulltraj[1]  # m
    setpoint.position.z = fulltraj[2]  # m
    setpoint.velocity.x = fulltraj[3]  # m/s
    setpoint.velocity.y = fulltraj[4]  # m/s
    setpoint.velocity.z = fulltraj[5]  # m/s
    setpoint.acceleration.x = fulltraj[6]  # m/s^2
    setpoint.acceleration.y = fulltraj[7]  # m/s^2
    setpoint.acceleration.z = fulltraj[8]  # m/s^2
    setpoint.attitude.yaw = 0  # deg
    return setpoint

def updateSensor(sensors,uavState):
    sensors.gyro.x = uavState[10] # deg/s
    sensors.gyro.y = uavState[11] # deg/s
    sensors.gyro.z = uavState[12] # deg/s
    return sensors

def updateState(state, uavState):
    state.position.x = uavState[0]   # m
    state.position.y = uavState[1]    # m
    state.position.z = uavState[2]    # m
    state.velocity.x = uavState[3]    # m/s
    state.velocity.y = uavState[4]    # m/s
    state.velocity.z = uavState[5]    # m/s
    q_curr = np.array(uavState[6:10]).reshape((4,))
    rpy_state  = to_euler(q_curr,convention='zyx')
    state.attitude.roll  = rpy_state[0]
    state.attitude.pitch = rpy_state[1]
    state.attitude.yaw   = rpy_state[2]
    return state


if __name__ == '__main__':
	main()



#   // qr = mkvec(radians(setpoint->attitude.roll),
#             //     radians(setpoint->attitude.pitch), // This is in the legacy coordinate system where pitch is inverted
#             //       radians(setpoint->attitude.yaw));
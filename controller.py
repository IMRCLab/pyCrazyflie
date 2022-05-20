import numpy as np
import rowan as rn
import matplotlib.pyplot as plt
from uavDy import uav
from uavDy.uav import skew
from Animator import animateSingleUav 
from trajectoriescsv import *
import time
import argparse
import sys

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

def initController():
    """This function initializes the controller"""
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
    """This function sets the trajectory modes of the controller"""
    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.mode.roll = cffirmware.modeDisable
    setpoint.mode.pitch = cffirmware.modeDisable
    setpoint.mode.yaw = cffirmware.modeDisable
    return setpoint

def updateDesState(setpoint, fulltraj):
    """This function updates the desired states"""
    setpoint.position.x = fulltraj[0]  # m
    setpoint.position.y = fulltraj[1]  # m
    setpoint.position.z = fulltraj[2]  # m
    setpoint.velocity.x = fulltraj[3]  # m/s
    setpoint.velocity.y = fulltraj[4]  # m/s
    setpoint.velocity.z = fulltraj[5]  # m/s
    setpoint.acceleration.x = fulltraj[6]  # m/s^2
    setpoint.acceleration.y = fulltraj[7]  # m/s^2
    setpoint.acceleration.z = fulltraj[8]  # m/s^2
    # setpoint.attitude.yaw = 0  # deg
    return setpoint
    
def updateSensor(sensors, uav):
    """This function updates the sensors signals"""
    uavState = uav.state
    sensors.gyro.x = np.degrees(uavState[10]) # deg/s
    sensors.gyro.y = np.degrees(uavState[11]) # deg/s
    sensors.gyro.z = np.degrees(uavState[12]) # deg/s
    return sensors

def updateState(state, uav):
    """This function passes the current states to the controller"""
    uavState = uav.state
    state.position.x = uavState[0]   # m
    state.position.y = uavState[1]    # m
    state.position.z = uavState[2]    # m
    state.velocity.x = uavState[3]    # m/s
    state.velocity.y = uavState[4]    # m/s
    state.velocity.z = uavState[5]    # m/s
    q_curr = np.array(uavState[6:10]).reshape((4,))
    rpy_state  = rn.to_euler(q_curr,convention='xyz')
    state.attitude.roll  = np.degrees(rpy_state[0])
    state.attitude.pitch = np.degrees(-rpy_state[1])
    state.attitude.yaw   = np.degrees(rpy_state[2])
    state.attitudeQuaternion.w = q_curr[0]
    state.attitudeQuaternion.x = q_curr[1]
    state.attitudeQuaternion.y = q_curr[2]
    state.attitudeQuaternion.z = q_curr[3]
    fullState = np.array([state.position.x,state.position.y,state.position.z, 
                          state.velocity.x,state.velocity.y, state.velocity.z, 
                          q_curr[0],q_curr[1],q_curr[2],q_curr[3], uavState[10],uavState[11],uavState[12]]).reshape((13,))
    return state, fullState

def initializeState(uav_params):
    """This function sets the initial states of the UAV
        dt: time step
        initPose: initial position [x,y,z]
        initq: [qw, qx, qy, qz] initial rotations represented in quaternions 
        initLinVel: [xdot, ydot, zdot] initial linear velocities
        initAngVel: [wx, wy, wz] initial angular velocities"""
    dt = float(uav_params['dt'])
    
    initPos = np.array(uav_params['init_pos_Q'])
    
    # initialize Rotation matrix about Roll-Pitch-Yaw
    attitude = uav_params['init_attitude_Q'] 
    for i in range(0,len(attitude)):
        attitude[i] = np.radians(attitude[i])
    initq = rn.from_euler(attitude[0],attitude[1],attitude[2])    
    
    #Initialize Twist
    initLinVel = np.array(uav_params['init_linVel_Q'])
    initAngVel = np.array(uav_params['init_angVel_Q'])
    ### State = [x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, wx, wy, wz] ###
    initState = np.zeros((13,))
    initState[0:3]  = initPos  # position: x,y,z
    initState[3:6]  = initLinVel  # linear velocity: xdot, ydot, zdot
    initState[6:10] = initq# quaternions: [qw, qx, qy, qz]
    initState[10::] = initAngVel # angular velocity: wx, wy, wz
    return dt, initState

def initializeStateWithPayload(payload_cond):
    """This function sets the initial states of the UAV-Payload system
        dt: time step
        initPose: initial payload position [xl,yl,zl]
        initLinVel: [xldot, yldot, zldot] initial linear velocities
        initp: initial directional unit vector pointing from UAV to payload expressed in Inertial frame
        initq: [qw, qx, qy, qz] initial rotations represented in quaternions 
        initAngVel: [wx, wy, wz] initial angular velocities"""

    dt = float(payload_cond['dt'])
    lc = float(payload_cond['l_c']) # length of cable [m] 
    
    initPosL = np.array(payload_cond['init_pos_L']) #  Initial position
    initp    = np.array(payload_cond['p']) #  Initial Unit vector

     #Initialize payload Twist
    inLinVL  = np.array(payload_cond['init_linV_L']) # Linear velocity of payload
    inAnVL   = np.array(payload_cond['wl']) # Angular Velocity of Payload
    
    # initialize Rotation matrix: Roll-Pitch-Yaw
    attitude = payload_cond['init_attitude_Q'] 
    for i in range(0,len(attitude)):
        attitude[i] = np.radians(attitude[i])
    initq = rn.from_euler(attitude[0],attitude[1],attitude[2])    

    # Initialize anglular velocity of quadrotor
    initAngVel = np.array(payload_cond['init_angVel_Q'])
    initState  = np.zeros((19,))

    initState[0:3]   = initPosL
    initState[3:6]   = inLinVL
    initState[6:9]   = initp
    initState[9:12]  = inAnVL
    initState[12:16] = initq
    initState[16::]  = initAngVel
    return dt, initState

def StQuadfromPL(payload):
    """This function initializes the states of the quadrotor given the states of the payload """ 
    uavState =  np.zeros((13,))
    posq = payload.state[0:3] - payload.lc * payload.state[6:9]
    pdot = np.cross(payload.state[9:12], payload.state[6:9])
    velq = payload.state[3:6] - payload.lc * pdot
    uavState[0:3]  = posq
    uavState[3:6]  = velq
    uavState[6:10] = payload.state[12:16]
    uavState[10::] =  payload.state[16::]
    return uavState

def animateTrajectory(uavs, payloads, videoname, shared):
    # Animation    
    fig     = plt.figure(figsize=(10,10))
    ax      = fig.add_subplot(autoscale_on=True,projection="3d")
    sample  = 100 
    animate = animateSingleUav.PlotandAnimate(fig, ax, uavs, payloads, sample, shared) 
    dt_sampled = list(uavs.values())[0].dt * sample
    print("Starting Animation... \nAnimating, Please wait...")
    now = time.time()
    startanimation = animate.startAnimation(videoname,dt_sampled)
    print("Converting Animation to Video. \nPlease wait...")

    end = time.time()
    plt.close(fig)
    print("Run time:  {:.3f}s".format((end - now)))

def animateOrPlot(uavs, payloads, animateOrPlotdict, filename, tf_sim, shared): 
    if animateOrPlotdict['animate']:
        videoname = filename + '.gif'
        animateTrajectory(uavs, payloads, videoname, shared)     
    # The plot will be shown eitherways
    # savePlot: saves plot in pdf format
    pdfName = filename + '.pdf'
    animateSingleUav.outputPlots(uavs, payloads, animateOrPlotdict['savePlot'], tf_sim, pdfName, shared)

def setParams(params):
    dt           = float(params['dt'])
    uavs, payloads, trajectories  = {}, {}, {}
    
    for name, robot in params['Robots'].items():
        trajectories['uav_'+name]   = robot['refTrajPath']
        if robot['payload']['mode'] in 'enabled':
            payload_params          = {**robot['payload'], **robot['initConditions'], 'm':robot['m'], 'dt':dt}
            dt, initState           = initializeStateWithPayload(payload_params)
            payload                 = uav.Payload(dt, initState, payload_params)
            uav1                    = uav.UavModel(dt, StQuadfromPL(payload), robot, pload=True, lc=payload.lc)
            uavs['uav_'+name]       = uav1
            payloads['uav_'+name] = payload
        else:
            uav_params     = {'dt': dt, **robot['initConditions'], **robot}
            dt, initState  = initializeState(uav_params)
            uav1           = uav.UavModel(dt, initState, uav_params) 
            uavs['uav_'+name] = uav1
    return uavs, payloads, trajectories        

def StatefromSharedPayload(payload, angState, lc, j):
    ## Thid method computes the initial conditions of each quadrotor
    #  given the initial condition of the payload and the directional unit vectors of each cable
    qi = payload.state[j:j+3]
    wi = payload.state[j+3*payload.numOfquads:j+3+3*payload.numOfquads]
    uavState =  np.zeros((13,))
    posq =  payload.state[0:3] - lc * qi
    pdot = np.cross(wi, qi)
    velq = payload.state[3:6] - lc * pdot
    uavState[0:3]  = posq
    uavState[3:6]  = velq
    uavState[6:10] = angState[0:4]
    uavState[10::] =  angState[4:]
    return uavState


def setPayloadfromUAVs(uavs_params, payload_params):
    ## THIS IS NOT USED NOW!!
    ## This method sets the states of the payload given the positions of the quadrotor. 
    ## This is opposite to what is normally done, but since the controller for the whole system
    ## has not been yet finished, then provided an initial condition of all UAVs and the length of the cables,
    ## the payload initial conditions are computed. it is activated through --initUavs flag argument
    for params in uavs_params.values():
        posq = np.array(params['init_pos_Q'])
        lc   = params['l_c']
        vq   = np.array(params['init_linVel_Q'])
        angR = np.radians(params['q_dg'])
        q    =  rn.to_matrix(rn.from_euler(angR[0], angR[1], angR[2], convention='xyz',axis_type='extrinsic')) @ np.array([0,0,-1]) #
        qdot = np.array(params['qd'])
        initPos  = posq + lc * q
        initLinV = vq + lc * qdot
        print('q: ',q) 
        print('pload init pos: ',initPos)
        # print('pos init of quad: ', posq)
        # print('pload init vel: ',initLinV)
    payload_params.update({'init_pos_L': initPos, 'init_linV_L': initLinV})
    return payload_params, uav.SharedPayload(payload_params, uavs_params)
    pass

def setTeamParams(params, initUavs):
    dt    = float(params['dt'])
    uavs, trajectories = {}, {}
    plStSize = 13 # 13 is the number of the payload states.
            #  We want to get the angles and its derivatives
            #  between load and UAVs (Check the state structure of SharedPayload object)
    inertia = np.diag(np.array(params['RobotswithPayload']['payload']['inertia']))
    if np.linalg.det(inertia) == 0:
            plStSize -= 7 # if the payload is considered as a point mass than we only have the linear terms 
                          # thus the state: [xp, yp, zp, xpdot, ypdot, zpdot]
    ## --initUavs: this flag let us initialize the conditions of the payload, given the initial condtions
    ## of the UAVs (which is not what is normally done, but for the sake of having easier tests).
    if not initUavs:
        payload_params = {**params['RobotswithPayload']['payload'], 'dt': dt}
        uavs_params = {}
        for name, robot in params['RobotswithPayload']['Robots'].items():
            trajectories['uav_'+name]   = robot['refTrajPath']
            uavs_params.update({name: {**robot}})
        payload = uav.SharedPayload(payload_params, uavs_params)
        j = plStSize
        for name, robot in uavs_params.items():
            lc     = robot['l_c']
            eulAng = robot['initConditions']['init_attitude_Q']
            quat   = rn.from_euler(eulAng[0], eulAng[1], eulAng[2])
            w_i    = robot['initConditions']['init_angVel_Q']
            angSt  = np.hstack((quat, w_i)).reshape((7,))
            uav1   = uav.UavModel(dt, StatefromSharedPayload(payload, angSt, lc, j), robot, pload=True, lc=lc)
            j +=3
            uavs['uav_'+name] = uav1    
    else:
        payload_params = {**params['RobotswithPayload']['payload'], 'dt': dt}
        uavs_params    = {}
        for name, robot in params['RobotswithPayload']['Robots'].items():
            trajectories['uav_'+name]   = robot['refTrajPath']
            uavs_params.update({name: {**robot['initConditions'], **robot, 'dt': dt}})
            dt, initState  = initializeState(uavs_params[name])
            uav1           = uav.UavModel(dt, initState, uavs_params[name])
            uavs['uav_'+name] = uav1
        payload_params, payload = setPayloadfromUAVs(uavs_params, payload_params)
    return plStSize, uavs, uavs_params, payload, trajectories
##----------------------------------------------------------------------------------------------------------------------------------------------------------------##        
##----------------------------------------------------------------------------------------------------------------------------------------------------------------##
def main(filename, initUavs, animateOrPlotdict, params):
    # Initialize an instance of a uav dynamic model with:
    # dt: time interval
    # initState: initial state
    # set it as 1 tick: i.e: 1 ms
    # pload: payload flag, enabled: with payload, otherwise: no payload 
    shared = False
    if params['RobotswithPayload']['payload']['mode'] in 'shared':
       plStSize, uavs, uavs_params, payload, trajectories = setTeamParams(params, initUavs)
       shared = True
    else:
        uavs, payloads, trajectories = setParams(params)
    # Upload the traj in csv file format
    # rows: time, xdes, ydes, zdes, vxdes, vydes, vzdes, axdes, aydes, azdes  
    timeStamped_traj = {}
    if not uavs:
         sys.exit('no UAVs')
    for id in uavs.keys():
        input = trajectories[id]
        timeStamped_traj[id] = np.loadtxt(input, delimiter=',') 
        tf_ms = timeStamped_traj[id][0,-1]*1e3
    # Simulation time
    tf_sim = tf_ms + 0.5e3
      # final time of traj in ms
    print('\nTotal trajectory time: '+str(tf_sim*1e-3)+ 's')
    print('Simulating...')
    control, setpoint, sensors, state = initController()
    
    if params['RobotswithPayload']['payload']['mode'] in 'shared':
        for tick in range(0, int(tf_sim)+1):
            quadNum = 0
            j = plStSize
            ctrlStack = np.empty((1,4))
            for id in uavs.keys():
            #initialize the controller and allocate current state (both sensor and state are the state)
            # This is kind of odd and should be part of state
                if tick <= int(tf_ms):    
                    setpoint  = updateDesState(setpoint, timeStamped_traj[id][1::,tick])
                    ref_state =  timeStamped_traj[id][1:7,tick]
                else:
                    setpoint  = updateDesState(setpoint, timeStamped_traj[id][1::,-1])
                    ref_state = timeStamped_traj[id][1:7,-1]
                 # update current state
                state, fullState = updateState(state, uavs[id])
                sensors          = updateSensor(sensors, uavs[id])
                cffirmware.controllerSJC(control, setpoint, sensors, state, tick)

                control_inp = np.array([control.thrustSI, control.torque[0], control.torque[1], control.torque[2]])
                ctrlStack   = np.vstack((ctrlStack, control_inp.reshape(1,4)))
                ctrlInp     = control_inp[0]*rn.to_matrix(uavs[id].state[6:10]) @ np.array([0,0,1])
                              
                payload.stackCtrl(ctrlInp.reshape(1,3))

                uavs[id].state = StatefromSharedPayload(payload, fullState[6::], uavs[id].lc, j)
                uavs[id].stackStandCtrl(uavs[id].state, control_inp, ref_state)
                j +=3
            payload.cursorUp() 
            uavs, loadState =  payload.stateEvolution(ctrlStack, uavs, uavs_params)
            payload.stackState()
        payload.cursorPlUp()
        for id in uavs.keys():
            uavs[id].cursorUp()
        animateOrPlot(uavs, payload, animateOrPlotdict, filename, tf_sim, shared)

    else:
        for id, uav_ in uavs.items():
            #initialize the controller and allocate current state (both sensor and state are the state)
            # This is kind of odd and should be part of state
            control, setpoint, sensors, state = initController()
            # Note that 1 tick == 1ms
            # note that the attitude controller will only compute a new output at 500 Hz
            # and the position controller only at 100 Hz
            # If you want an output always, simply select tick==0
            if uav_.pload:
                payload = payloads[id]
            
            for tick in range(0, int(tf_sim)+1):
                # update desired state
                if tick <= int(tf_ms):    
                    setpoint  = updateDesState(setpoint, timeStamped_traj[id][1::,tick])
                    ref_state =  timeStamped_traj[id][1:7,tick]
                else:
                    setpoint  = updateDesState(setpoint, timeStamped_traj[id][1::,-1])
                    ref_state = timeStamped_traj[id][1:7,-1]
                # update current state
                state,fullState = updateState(state, uav_)
                sensors         = updateSensor(sensors, uav_)
                # query the controller
                cffirmware.controllerSJC(control, setpoint, sensors, state, tick)
                # states evolution
                control_inp = np.array([control.thrustSI, control.torque[0], control.torque[1], control.torque[2]])

                if uav_.pload:
                    payload.PL_nextState(control_inp, uav_)
                    uav_.state = StQuadfromPL(payload)
                else:
                    uav_.states_evolution(control_inp)
                uav_.stackStandCtrl(uav_.state, control_inp, ref_state)    
            uav_.cursorUp()
            uavs[id] = uav_
            if uav_.pload:
                payload.cursorUp()
                payloads[id] = payload
        # Animation        
        animateOrPlot(uavs, payloads, animateOrPlotdict, filename, tf_sim, shared)    


if __name__ == '__main__':
    try: 
        import cffirmware
        parser = argparse.ArgumentParser()
        parser.add_argument('filename', type=str, help="Name of the CSV file in trajectoriescsv directory")
        parser.add_argument('--animate', default=False, action='store_true', help='Set true to save a gif in Videos directory')
        parser.add_argument('--savePlot', default=False, action='store_true', help='Set true to save plots in a pdf  format')
        parser.add_argument('--initUavs', default=False, action='store_true', help='Set true to initialize the conditions of the UAVs and then compute the payload initial condition')
        args   = parser.parse_args()   
        animateOrPlotdict = {'animate':args.animate, 'savePlot':args.savePlot}
    
        import yaml
        with open('config/initialize.yaml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        main(args.filename, args.initUavs, animateOrPlotdict, params)
    except ImportError as imp:
        print(imp)
        print('Please export crazyflie-firmware/ to your PYTHONPATH')
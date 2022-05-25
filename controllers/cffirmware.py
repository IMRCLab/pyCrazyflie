import numpy as np 
import rowan as rn
from uavDy import uav
modeAbs     = 0
modeDisable = 1
modeVelocity = 2

class vec3_s:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None

class attitude_t:
    def __init__(self):
        self.roll  = None
        self.pitch = None
        self.yaw   = None

class quaternion_t:
    def __init__(self):
        self.q0 = None
        self.q1 = None
        self.q2 = None
        self.q3 = None
        self.x  = None
        self.y  = None
        self.z  = None
        self.w  = None

class mode:
    def __init__(self):
        self.x     = None
        self.y     = None
        self.z     = None
        self.roll  = None
        self.pitch = None
        self.roll  = None
        self.yaw   = None
        self.quat  = None

class setpoint_t:
    def __init__(self): 
        self.position = vec3_s()
        self.attitude = attitude_t()
        self.attitudeQuaternion = quaternion_t()
        self.velocity = vec3_s()
        self.attitudeRate = attitude_t()
        self.acceleration = vec3_s()
        self.mode = mode() 

class baro_t:
    def __init__(self):
        self.pressure = None  
        self.temperature = None
        self.asl = None

class sensorData_t:
    def __init__(self):
        self.acc = vec3_s()  # Gs
        self.gyro = vec3_s() # deg/s
        self.mag = vec3_s()  #gauss
        self.baro = baro_t();

class control_t:
    def __init__(self):
        self.roll  = 0
        self.pitch = 0
        self.yaw   = 0
        self.normalizedForces = 0
        
        self.thrustSI = 0
        self.torque = np.array([0,0,0])
        self.controlMode = None

class state_t:
    def __init__(self):
        self.position = vec3_s()
        self.attitude = attitude_t()
        self. attitudeQuaternion = quaternion_t()
        self.velocity = vec3_s()
        self.acc = vec3_s()

def controllerLeeInit():
    pass

def controllerLeeReset():
    pass

def computeDesiredRot(Fd, yaw):
    Rd = np.eye(3)
    normFd = np.linalg.norm(Fd)
    # print(normFd)
    if normFd > 0:
        zdes = (Fd/normFd).reshape(3,)
        # print(zdes)
    else:
      zdes = np.array([0,0,1])  
    xdes = np.array([np.cos(yaw), np.sin(yaw), 0])
    normZX = np.linalg.norm(uav.skew(zdes) @ xdes)
    if normZX > 0:
        ydes = ((uav.skew(zdes)@xdes)/(normZX))
    else:
        ydes = np.arange([0,1,0])
    Rd[:,0] = xdes
    Rd[:,1] = ydes
    Rd[:,2] = zdes
    return Rd

def flatten(w_tilde):
    w1 = w_tilde[2,1]
    w2 = w_tilde[0,2]
    w3 = w_tilde[1,0]
    return np.array([w1,w2,w3])
    
def controllerLee(uavModel, control, setpoint, sensors, state, tick):
    
    Kp      = uavModel.controller['kp'] * np.identity(3)
    Kv      = uavModel.controller['kd'] * np.identity(3)
    Kw      = uavModel.controller['kw'] * np.identity(3)
    Kr      = uavModel.controller['kr'] * np.identity(3)
    currPos = np.array([state.position.x, state.position.y, state.position.z])
    currVl  = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
    
    desPos = np.array([setpoint.position.x, setpoint.position.y, setpoint.position.z])
    desVl  = np.array([setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z])    
    desAcc = np.array([setpoint.acceleration.x, setpoint.acceleration.y, setpoint.acceleration.z])
    ep = (desPos - currPos).reshape((3,1))
    ev = (desVl   - currVl).reshape((3,1))
    m = uavModel.m
    I = uavModel.I
    
    gravComp = np.array([0,0,m*9.81]).reshape((3,1))
    R = rn.to_matrix(uavModel.state[6:10])

    Rt = np.transpose(R)
    Fd = ((desAcc + gravComp + Kv @ ev + Kp @ ep) @ R @ np.array([0,0,1])).reshape((3,1))
    Rd  = computeDesiredRot(Fd, setpoint.attitude.yaw)
    Rtd = np.transpose(Rd)
    er      = flatten(0.5* (Rt @ Rd - Rtd @ R)).reshape((3,1))
    curr_w  = uavModel.state[10::].reshape((3,1))
    des_w   = np.zeros((3,1))
    ew      = (Rt @ Rd @ des_w - curr_w).reshape((3,1))
    tau = (Kr @ er  + Kw @ ew + uav.skew(curr_w)@I@curr_w - I @ (uav.skew(curr_w) @ Rt @ Rd @ des_w)).reshape((3,1))
   
    control.thrustSI = np.linalg.norm(Fd)
    control.torque = tau.reshape(3,)
    return control

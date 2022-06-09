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
        self.jerk = vec3_s()
        self.snap = vec3_s()
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
    if normFd > 0:
        zdes = (Fd/normFd).reshape(3,)
    else:
      zdes = np.array([0,0,1])  
    xcdes = np.array([np.cos(yaw), np.sin(yaw), 0])
    normZX = np.linalg.norm(uav.skew(zdes) @ xcdes)
    if normZX > 0:
        ydes = ((np.cross(zdes.reshape(3,), xcdes))/(normZX))
    else:
        ydes = np.array([0,1,0])
    xdes = np.cross(ydes.reshape(3,), zdes.reshape(3,))
    Rd[:,0] = xdes.reshape(3,)
    Rd[:,1] = ydes.reshape(3,)
    Rd[:,2] = zdes.reshape(3,)
    return Rd

def flatten(w_tilde):
    w1 = w_tilde[2,1]
    w2 = w_tilde[0,2]
    w3 = w_tilde[1,0]
    return np.array([w1,w2,w3])

def computeWd(m, R, T, desjerk):
    xb = R[:,0]
    yb = R[:,1]
    zb = R[:,2]
    if T == 0:
        hw = np.zeros(3,)
    else:
        hw = m/T * (desjerk - np.dot(zb, desjerk)*zb)
    p  = -np.dot(hw, yb)
    q  = np.dot(hw, xb)
    r  = 0
    return np.array([p,q,r])

def computeWddot(m, R, curr_w, T, Td, Td_dot, dessnap):
    xb = R[:,0]
    yb = R[:,1]
    zb = R[:,2]
    curr_w = curr_w.reshape(3,)
    if T == 0: 
        ha = np.zeros(3,)
    else:
        ha = (m/T)*dessnap - np.dot((Td_dot/T), zb) - (2/T)*np.cross(curr_w, np.dot(Td, zb)) \
            - np.cross(np.cross(curr_w, curr_w), zb) 
    return np.array([-np.dot(ha, yb), np.dot(ha, xb), 0])

def controllerLee(uavModel, control, setpoint, sensors, state, tick):
    
    kp      = uavModel.controller['kp']
    kv      = uavModel.controller['kd']
    kw      = uavModel.controller['kw']
    kr      = uavModel.controller['kr']
    
    currPos = np.array([state.position.x, state.position.y, state.position.z]).reshape((3,1))
    currVl  = np.array([state.velocity.x, state.velocity.y, state.velocity.z]).reshape((3,1))
    desPos = np.array([setpoint.position.x, setpoint.position.y, setpoint.position.z]).reshape((3,1))
    desVl  = np.array([setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z]).reshape((3,1))    
    desAcc = np.array([setpoint.acceleration.x, setpoint.acceleration.y, setpoint.acceleration.z]).reshape((3,1))
    desjerk = np.array([setpoint.jerk.x, setpoint.jerk.y, setpoint.jerk.z]).reshape((3,))
    dessnap = np.array([setpoint.snap.x, setpoint.snap.y, setpoint.snap.z]).reshape((3,))
    ep = (currPos - desPos)
    ev = (currVl  - desVl)
    m  = uavModel.m
    I  = uavModel.I
    
    gravComp = np.array([0,0,9.81]).reshape((3,1))
    R = rn.to_matrix(uavModel.state[6:10])
    Rt = np.transpose(R)

    FdI = desAcc + gravComp  - kp * ep - kv * ev
    control.thrustSI  = (m * FdI.T @ R @ np.array([[0],[0],[1]]))[0,0]
    # print(ep.T, tick)
    Rd  = computeDesiredRot(FdI,0)
    Rtd = np.transpose(Rd)
    
    er       = 0.5 * flatten((Rtd @ R - Rt @ Rd)).reshape((3,1)) 
    curr_w   = uavModel.state[10::].reshape((3,1))
    curr_w_  = curr_w.reshape(3,) # reshape of omega for cross products

    zb      = Rd[:,2]
    T       = control.thrustSI#m * np.dot(FdI.reshape(3,), zb)
    Td      = m * np.dot(desjerk, zb)
    des_w  = (computeWd(m, Rd, T, desjerk)).reshape((3,1))
    des_w_ = des_w.reshape(3,)
    Td_dot  = np.dot(zb, m * dessnap) - np.dot(zb, np.cross(np.cross(des_w_, des_w_), np.dot(T, zb)))
    des_wd  = (computeWddot(m, R, des_w, T, Td, Td_dot, dessnap)).reshape(3,1)
    ew  = (curr_w - Rt @ Rd @ des_w).reshape((3,1))

    control.torque = ( -kr * er  - kw * ew + (np.cross(curr_w_, (I @ curr_w_))).reshape(3,1) \
        - I @ (uav.skew(curr_w) @ Rt @ Rd @ des_w - Rt @ Rd @ des_wd) ).reshape(3,)
    return control, des_w.reshape(3,), des_wd.reshape(3,)

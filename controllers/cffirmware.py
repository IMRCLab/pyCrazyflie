import numpy as np 
import rowan as rn
from uavDy import uav
from scipy import linalg as la
import sys
import cvxpy as cp
import osqp 
from scipy import sparse
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
        self.u_all = np.zeros(3,)
        self.controlMode = None

class state_t:
    def __init__(self):
        self.position = vec3_s()
        self.attitude = attitude_t()
        self. attitudeQuaternion = quaternion_t()
        self.velocity = vec3_s()
        self.acc = vec3_s()
        self.payload_pos = vec3_s()
        self.payload_vel = vec3_s()

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

def thrustCtrl(m, R ,refAcc, kpep, kvev):
    FdI = refAcc - kpep - kvev
    return (m * FdI.T @ R @ np.array([[0],[0],[1]]))[0,0], FdI

def torqueCtrl(I, Rt, curr_w_, krer, kwew, Rd, des_w, des_wd):
    curr_w = curr_w_.reshape((3,1))
    return ( -krer  - kwew + (np.cross(curr_w_, (I @ curr_w_))).reshape(3,1) \
        - I @ (uav.skew(curr_w) @ Rt @ Rd @ des_w - Rt @ Rd @ des_wd) ).reshape(3,)

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
    control.thrustSI, FdI = thrustCtrl(m, R, desAcc+gravComp, kp*ep, kv*ev)

    Rd  = computeDesiredRot(FdI,0)
    Rtd = np.transpose(Rd)
    
    er       = 0.5 * flatten((Rtd @ R - Rt @ Rd)).reshape((3,1)) 
    curr_w   = uavModel.state[10::].reshape((3,1))
    curr_w_  = curr_w.reshape(3,) # reshape of omega for cross products

    zb      = Rd[:,2]
    T       = control.thrustSI
    Td      = m * np.dot(desjerk, zb)
    des_w  = (computeWd(m, Rd, T, desjerk)).reshape((3,1))
    des_w_ = des_w.reshape(3,)
    Td_dot  = np.dot(zb, m * dessnap) - np.dot(zb, np.cross(np.cross(des_w_, des_w_), np.dot(T, zb)))
    des_wd  = (computeWddot(m, R, des_w, T, Td, Td_dot, dessnap)).reshape(3,1)
    ew  = (curr_w - Rt @ Rd @ des_w).reshape((3,1))

    control.torque = torqueCtrl(I, Rt, curr_w_, kr*er, kw*ew, Rd, des_w, des_wd)
    
    return control, des_w.reshape(3,), des_wd.reshape(3,)

def controllerLeePayloadInit():
    pass

def torqueCtrlwPayload(uavModel, fi, payload, setpoint, tick):
    kwx     = float(payload.controller['kwx'])
    kwy     = float(payload.controller['kwy'])
    kwz     = float(payload.controller['kwz'])
    kw = np.diag(np.array([kwx, kwy, kwz]))
    krx     = float(payload.controller['krx'])
    kry     = float(payload.controller['kry'])
    krz     = float(payload.controller['krz'])
    kr = np.diag(np.array([krx, kry, krz])) 
    
    desjerk = np.array([setpoint.jerk.x, setpoint.jerk.y, setpoint.jerk.z]).reshape((3,))
    dessnap = np.array([setpoint.snap.x, setpoint.snap.y, setpoint.snap.z]).reshape((3,))
    R = rn.to_matrix(uavModel.state[6:10])
    Rt = R.T
    u = fi
    m = uavModel.m
    I = uavModel.I
    Rd =  computeDesiredRot(u,0)
    Rtd = np.transpose(Rd)
    psi = 0.5*np.trace(np.eye(3)-Rtd@R)
    er       = 0.5 * flatten((Rtd @ R - Rt @ Rd)).reshape((3,1)) 
    curr_w   = uavModel.state[10::].reshape((3,1))
    curr_w_  = curr_w.reshape(3,) # reshape of omega for cross products
    zb      = Rd[:,2]
    T       = np.linalg.norm(fi)
    Td      = m * np.dot(desjerk, zb)
    des_w  = (computeWd(m, Rd, T, desjerk)).reshape((3,1))
    des_w_ = des_w.reshape(3,)
    Td_dot  = np.dot(zb, m * dessnap) - np.dot(zb, np.cross(np.cross(des_w_, des_w_), np.dot(T, zb)))
    des_wd  = (computeWddot(m, R, des_w, T, Td, Td_dot, dessnap)).reshape(3,1)
    ew  = (curr_w - Rt @ Rd @ des_w).reshape((3,1))
    torques =  torqueCtrl(I, Rt, curr_w_, kr@er, kw@ew, Rd, des_w, des_wd)   

    return torques, des_w, des_wd

def parallelComp(virtualInp, uavModel, payload, j):
    ## This only includes the point mass model
    grav = np.array([0,0,-9.81])
    acc_ = payload.accl[0:3] #(payload.state[3:6] - payload.prevSt[3:6])/payload.dt
    acc0 = acc_ - grav
    m   = uavModel.m
    l = uavModel.lc
    qi = payload.state[j:j+3]
    wi = payload.state[j+3*payload.numOfquads:j+3+3*payload.numOfquads]   
    qiqiT = qi.reshape((3,1))@(qi.T).reshape((1,3))
    u_parallel = virtualInp + m*l*(np.dot(wi, wi))*qi  +  m*qiqiT@acc0
    return u_parallel

def perpindicularComp(desVirtInp, uavModel, payload, kq, kw, ki, j, tick):
    ## This only includes the point mass model
    grav = np.array([0,0,-9.81])
    acc_ = payload.accl[0:3] #(payload.state[3:6] - payload.prevSt[3:6])/payload.dt
    acc0 = acc_ - grav
    qdi    = - desVirtInp/ np.linalg.norm(desVirtInp)
    if tick == 0: 
        payload.qdi_prevdict[uavModel.id] = qdi.copy()
    qdidot = (qdi - payload.qdi_prevdict[uavModel.id])/payload.dt
    payload.qdi_prevdict[uavModel.id] = qdi.copy()
    
    m   = uavModel.m
    l = uavModel.lc
    qi = payload.state[j:j+3]
    wi = payload.state[j+3*payload.numOfquads:j+3+3*payload.numOfquads]   
    qidot = np.cross(wi, qi)
    
    # P part
    eq = np.cross(qdi, qi)

    wdi = np.cross(qdi, qdidot)
    skewqi2 = (uav.skew(qi)@uav.skew(qi))
    # D part
    ew = wi  + skewqi2 @ wdi

    if tick == 0: 
        payload.wdi_prevdict[uavModel.id] = wdi.copy()
    wdidot = (wdi - payload.wdi_prevdict[uavModel.id])/payload.dt
    payload.wdi_prevdict[uavModel.id] = wdi.copy()

    qiqiT = qi.reshape((3,1))@(qi.T).reshape((1,3))
    u_perp = m * l  * uav.skew(qi) @ (-kq @ eq - kw @ ew- np.dot(qi, wdi)*qidot - skewqi2@wdidot) - m * skewqi2 @ acc0 
    return u_perp

def controllerLeePayload(uavs, id, payload, control, setpoint, sensors, state, tick, j):
    uavModel = uavs[id]
    kpx      = float(payload.controller['kpx'])
    kpy      = float(payload.controller['kpy'])
    kpz      = float(payload.controller['kpz'])
    kp = np.diag(np.array([kpx, kpy, kpz]))
    kdx      = float(payload.controller['kdx'])
    kdy      = float(payload.controller['kdy'])
    kdz      = float(payload.controller['kdz'])
    kd = np.diag(np.array([kdx, kdy, kdz]))
    kipx    = float(payload.controller['kipz'])
    kipy    = float(payload.controller['kipy'])
    kipz    = float(payload.controller['kipz'])
    ki_p = np.diag(np.array([kipx, kipy, kipz]))
  
    kqx      = float(payload.cablegains['kqx'])
    kqy      = float(payload.cablegains['kqy'])
    kqz      = float(payload.cablegains['kqz'])
    kq   = np.diag(np.array([kqx, kqy, kqz]))
    kwcx     = float(payload.cablegains['kwcx'])
    kwcy     = float(payload.cablegains['kwcy'])
    kwcz     = float(payload.cablegains['kwcz'])
    kwc = np.diag(np.array([kwcx, kwcy, kwcz]))
    ki  = np.diag(np.array([0,0,0]))
    
    kwpx     = float(payload.controller['kwpx'])
    kwpy     = float(payload.controller['kwpy'])
    kwpz     = float(payload.controller['kwpz'])
    kwp = np.diag(np.array([kwpx, kwpy, kwpz]))
    krpx     = float(payload.controller['krpx'])
    krpy     = float(payload.controller['krpy'])
    krpz     = float(payload.controller['krpz'])

    krp = np.diag(np.array([krpx, krpy, krpz])) 

    currPos = np.array([state.payload_pos.x, state.payload_pos.y, state.payload_pos.z]).reshape((3,1))
    currVl  = np.array([state.payload_vel.x, state.payload_vel.y, state.payload_vel.z]).reshape((3,1))
    desPos  = np.array([setpoint.position.x, setpoint.position.y, setpoint.position.z]).reshape((3,1))
    desVl   = np.array([setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z]).reshape((3,1))    
    desAcc  = np.array([setpoint.acceleration.x, setpoint.acceleration.y, setpoint.acceleration.z]).reshape((3,1))
    desjerk = np.array([setpoint.jerk.x, setpoint.jerk.y, setpoint.jerk.z]).reshape((3,))
    dessnap = np.array([setpoint.snap.x, setpoint.snap.y, setpoint.snap.z]).reshape((3,))
    
    ep = (currPos - desPos)
    ev = (currVl  - desVl)
    payload.i_error = payload.i_error.reshape((3,1)) + payload.dt * ep
    ei = payload.i_error.reshape((3,1))
    mp  = payload.mp 
    gravComp = np.array([0,0,9.81]).reshape((3,1))

    Fd = mp * (- kp @ ep - kd @ ev + desAcc + gravComp)

    qi = payload.state[j:j+3]
    wi = payload.state[j+3*payload.numOfquads:j+3+3*payload.numOfquads]

    Md = np.zeros(3,)
    Rp = np.eye(3)
    rows = 3
    Rdp = np.eye(3)
    wdp = np.zeros(3,)
    Ud = Fd.copy()

    if not payload.pointmass: 
        Rp = rn.to_matrix(payload.state[6:10])
        rows = 6
        erp = 0.5* flatten(Rdp.T @ Rp - Rp.T @ Rdp)
        ewp = payload.state[10:13] - Rp.T @ Rdp @ wdp
        Md = -krp@erp - kwp@ewp #+ uav.skew(Rp.T @ Rdp @ wdp) @ payload.J @ Rp.T @ Rdp @ wdp 
        Ud = np.array([Rp.T @ Fd.reshape(3,), Md]).reshape(6,)
    quadNums = payload.numOfquads
    P = np.zeros((rows, 3*quadNums))
    k = 0
    R_p_diag = Rp
    for i in range(0,quadNums*3,3):
        P[0:3,i:i+3] = np.eye(3)
        if i >= 1:
            R_p_diag  = la.block_diag(R_p_diag, Rp)
        if not payload.pointmass:
            P[3::,i:i+3] = uav.skew(payload.posFrload[k,:]) 
            k+=1
    if payload.optimize:
        desVirtInp = qp(uavs, payload, Ud.reshape(rows,), P)
    else:
        P_inv = P.T @ np.linalg.inv(P@P.T)
        desVirtInp = (R_p_diag) @ (P_inv) @ (Ud.reshape(rows,))
    if not payload.pointmass:
        desVirtInp = desVirtInp[j-6-7:j-3-7]
    else:   
        desVirtInp = desVirtInp[j-6:j-3]    
    qiqiT = qi.reshape((3,1))@(qi.T).reshape((1,3))
    virtualInp =  qiqiT @ desVirtInp  
    u_parallel = parallelComp(virtualInp, uavModel, payload, j)
    u_perpind  = perpindicularComp(desVirtInp, uavModel, payload, kq, kwc, ki, j, tick)
    control.u_all = u_parallel + u_perpind

    R = rn.to_matrix(uavModel.state[6:10])
    Re3 = R@np.array([0,0,1])
    control.thrustSI = np.linalg.norm(control.u_all)
    torquesTick, des_w, des_wd = torqueCtrlwPayload(uavModel, control.u_all, payload, setpoint, tick*1e-3)
    control.torque = np.array([torquesTick[0], torquesTick[1], torquesTick[2]])

    return control, des_w, des_wd

def qlimit(uavs, payload, numofquads):
    # q_limit = np.empty((numofquads,3))
    # q1 = np.radians([-20 ,  0, 0]) 
    # q2 = np.radians([20, 0, 0])

    # q1 = rn.to_matrix(rn.from_euler(q1[0], q1[1], q1[2], convention='xyz',axis_type='extrinsic')) @ np.array([0,0,-1])
    # q2 = rn.to_matrix(rn.from_euler(q2[0], q2[1], q2[2], convention='xyz',axis_type='extrinsic')) @ np.array([0,0,-1])
    # # Rotate the vectors by 20 degrees
    # q_rotate = np.radians([0,0,20])
    # q_rotate = rn.from_euler(q_rotate[0], q_rotate[1], q_rotate[2],convention='xyz',axis_type='extrinsic')

    # q1_ = rn.rotate(q_rotate, q1)
    # q2_ = rn.rotate(q_rotate, q2)
  
    # n1 = np.cross(q1, q1_)
    # n2 = np.cross(q2, q2_)

    # A = np.zeros((2,6))
    # A[0, 0:3] = n1
    # A[1, 3:6] = n2
    n_stack = np.empty((1,3))
    for id in uavs.keys():
        hrpy = np.radians(uavs[id].hyperrpy)
        hyaw = np.radians(uavs[id].hyperyaw)     
        for i in range(len(hrpy)):
            hrpyi = hrpy[i]
            x = np.array([np.cos(hyaw[i][0]), np.sin(hyaw[i][0]), 0]).reshape(3,)
            z = rn.to_matrix(rn.from_euler(hrpyi[0], hrpyi[1], hrpyi[2], convention='xyz', axis_type='extrinsic')) @np.array([0,0,1])
            normZX = np.linalg.norm(np.cross(z, x))
            ni = ((np.cross(x, z))/(normZX))
            n_stack = np.vstack((n_stack, ni.reshape(1,3)))
    n_stack = np.delete(n_stack,  0, 0)
    hplanesNums = n_stack.shape[0]
    A = np.zeros((hplanesNums, 3*numofquads))
    j, k = 0, 0
    for id in uavs.keys():
        for i in range(len(uavs[id].hyperrpy)):
            A[k, j:j+3] = n_stack[k,:]
            k+=1 
        j+=3
    return A

def qp(uavs, payload, Ud, P_alloc):
    size = 3*payload.numOfquads
    P = np.eye(size)
    Ain = qlimit(uavs, payload, payload.numOfquads)
    if payload.qp_tool == 'cvxpy':
        mu_des = cp.Variable((size,))
        objective   = cp.Minimize((1/2)*cp.quad_form(mu_des, P))
        constraints = [P_alloc@mu_des == Ud,
                        Ain@mu_des <= np.zeros(Ain.shape[0],)]
        
        prob = cp.Problem(objective, constraints)
        data, chain, inverse_data = prob.get_problem_data(cp.OSQP)
        # print('data: ',data.keys())
        # for key in data.keys():
        #     print(key, '\n', data[key],'\n')
        prob.solve(verbose=False, solver='OSQP')
        mu_des = mu_des.value 
 
    elif payload.qp_tool == 'osqp':
        A     = sparse.vstack((P_alloc, Ain), format='csc') 
        P     = sparse.csc_matrix(P)
        q     = np.zeros(size)
        l     = np.hstack([Ud, -np.inf*np.ones(Ain.shape[0],)])
        u     = np.hstack([Ud, np.zeros(Ain.shape[0],)]) 
        prob = osqp.OSQP()
        # SAME SETTINGS AS CVXPY
        # settings = {'eps_abs': 1.0e-5, 'eps_rel' : 1.0e-05,
        #   'eps_prim_inf' : 1.0e-04, 'eps_dual_inf' : 1.0e-04,'rho' : 1.00e-01,
        #   'sigma' : 1.00e-06, 'alpha' : 1.60 ,'max_iter': 1000, 
        #   'verbose': False, 'linsys_solver': 'qdldl', 'check_termination': 25, 'polish': True}
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
        mu_des = prob.solve()
        mu_des = mu_des.x
    else:
        print('Please choose either cvxpy framework or osqp solver in the mode of the payloadCtrl')
        sys.exit()
    return mu_des
import numpy as np 
import rowan as rn
from uavDy import uav
from scipy import linalg as la
import sys
import cvxpy as cp
import osqp 
from scipy import sparse
from itertools import permutations, combinations, chain
modeAbs     = 0
modeDisable = 1
modeVelocity = 2
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

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

class hyperplane:
    def __str__(self):
      return  "normal: {}{}{}, offset:{} ".format(self.n[0], self.n[1], self.n[2], self.a)
    def __init__(self, n, a):
        self.n = n
        self.a = a
    def coeffs(self):
        return np.array([self.n[0], self.n[1], self.n[2], self.a])

class boundingBox:
    def __init__(self, p, dx, dy, dz):
        dx_v = np.array([dx, 0, 0]) # distance x from center
        dy_v = np.array([0, dy, 0]) # distance y from center
        dz_v = np.array([0, 0, dz]) # distance z from center
        self.cornerPoints = np.array([p+dx_v+dy_v+dz_v, p-dx_v+dy_v+dz_v, 
                p+dx_v-dy_v+dz_v, p-dx_v-dy_v+dz_v, p+dx_v+dy_v-dz_v,
                p-dx_v+dy_v-dz_v, p+dx_v-dy_v-dz_v, p-dx_v-dy_v-dz_v])


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

def normVec(n):
    normn = np.linalg.norm(n)
    if normn > 0:
        return n / np.linalg.norm(n)
    
    raise ValueError('norm of the vector is zero!')
    
def qlimit(uavs, payload, numofquads, tick):
    ids = list(uavs.keys())
    pairsinIds = list(permutations(ids, 2))
    n_stack = np.empty((1,3))
    a_s = []
    j = 0
    for pair in pairsinIds:   
        try:
            constraints = []
            pload = payload.state[0:3]
            pos1 = uavs[pair[0]].state[0:3] 
            pos2 = uavs[pair[1]].state[0:3] 
            n = cp.Variable(3,)
            a = cp.Variable()
            Q = np.eye(3)
            r = 0.1
            objective = cp.Minimize(cp.quad_form(n, Q))
            constraints.append([n.T@(pos1) - a <=  -1])
            constraints.append([n.T@(pos2) - a >=   1])
            constraints.append([n.T@pload  - a  ==  0])
            rpoint = pos1 + ((pos2-pos1)/2) - r*normVec((pos2-pos1))
            pdiff = rpoint - pos1
            print(pair, pdiff, pos1, rpoint)
            constraints.append([n.T@(pos1 + ((pos2-pos1)/2) - r*normVec((pos2-pos1)/2)) - a == 0])

            constraints = list(chain.from_iterable(constraints))        
            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=False, polish=True)            
            n_sol =  (n.value)           
            norm_n = np.linalg.norm(n_sol)
            if norm_n > 0:
                a_sol =  (np.array([a.value])[0])/norm_n
            else:
                raise ValueError('The norm of the vector is zero!')
            n_sol = normVec(n_sol)
            uavs[pair[0]].hp_prev[0:3] = n_sol
            uavs[pair[0]].hp_prev[3] = a_sol
            hp = hyperplane(n_sol, a_sol)
            n_stack = np.vstack((n_stack, n_sol.reshape(1,3)))
            a_s.append(a_sol)
            uavs[pair[0]].addHp(hp)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print('pair: ', pair)
            print('hp.normal: ',uavs[pair[0]].hp_prev[0:3], ' a:', uavs[pair[0]].hp_prev[3])
            print('hp.normal: ',uavs[pair[1]].hp_prev[0:3], ' a:', uavs[pair[1]].hp_prev[3])
            n_prev = uavs[pair[0]].hp_prev[0:3]
            a_prev = uavs[pair[0]].hp_prev[3]
            hp_prev = hyperplane(n_prev, a_prev)
            n_stack = np.vstack((n_stack, n_prev.reshape(1,3)))
            dist = pload.T@n_prev
            a_s.append(a_prev)
            uavs[pair[0]].addHp(hp_prev)
            raise
            break

    n_stack = np.delete(n_stack,  0, 0)
    hplanesNums = len(pairsinIds)
    A = np.zeros((hplanesNums, 3*numofquads))
    j, k = 0, 0    
    for id in uavs.keys():
        for i in range((numofquads-1)):
            A[k, j:j+3] = n_stack[k,:]
            k+=1 
        j+=3
        # print('pair:', pair, '\n',
        # 'nvec: ', n, '\n',
        # 'offset: ', hp.a)
    return uavs, A, a_s

def qp(uavs, payload, Ud, P_alloc, tick):
    size = 3*payload.numOfquads
    P = np.eye(size)
    uavs, Ain, a_s = qlimit(uavs, payload, payload.numOfquads, tick)
    try:
        if payload.qp_tool == 'cvxpy':
            mu_des = cp.Variable((size,))
            objective   = cp.Minimize((1/2)*cp.quad_form(mu_des, P))
            constraints = [P_alloc@mu_des == Ud,
                            Ain@mu_des + a_s <= np.zeros(Ain.shape[0]),]

            prob = cp.Problem(objective, constraints)
            # data, chain, inverse_data = prob.get_problem_data(cp.OSQP)
            # print('data: ',data.keys())
            # for key in data.keys():
            #     print(key, '\n', data[key],'\n')
            prob.solve(verbose=False, solver='OSQP')
            mu_des = mu_des.value 

        elif payload.qp_tool == 'osqp':
            A     = sparse.vstack((P_alloc, sparse.csc_matrix(Ain)), format='csc') 
            P     = sparse.csc_matrix(P)
            q     = np.zeros(size)
            l     = np.hstack([Ud, -np.inf*np.ones(Ain.shape[0],)])
            u     = np.hstack([Ud, np.negative(a_s)])
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
        payload.mu_des_prev = mu_des
        return uavs, payload, mu_des

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        print('mu_des: ', payload.mu_des_prev, '\n')
        mu_des = payload.mu_des_prev
        raise
        return uavs, payload, mu_des
        

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
    kipx    = float(payload.controller['kipx'])
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
        uavs, payload, desVirtInp = qp(uavs, payload, Ud.reshape(rows,), P, tick)
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

    return uavs, payload, control, des_w, des_wd

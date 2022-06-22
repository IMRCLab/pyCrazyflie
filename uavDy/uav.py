import numpy as np
from rowan.calculus import integrate as quat_integrate
from rowan import from_matrix, to_matrix, to_euler, from_euler
from scipy import  integrate, linalg
from numpy.polynomial import Polynomial as poly


def skew(w):
    w = w.reshape(3,1)
    w1 = w[0,0]
    w2 = w[1,0]
    w3 = w[2,0]
    return np.array([[0, -w3, w2],[w3, 0, -w1],[-w2, w1, 0]]).reshape((3,3))


class Payload:
    def __init__(self, dt, state, params):
        self.mp      = float(params['m_p']) # Mass of payload [kg]
        self.lc      = float(params['l_c'])  # length of cable [m]
        self.m       = float(params['m']) # Mass of quadrotor [kg]
        self.mt      = self.m + self.mp # Total mass [kg]
        self.grav_   = np.array([0,0,-self.mt*9.81])
        # state = [xl, yl, zl, xldot, yldot, zldot, px, py, pz, wlx, wly, wlz, qw, qx, qy, qz, wx, wy, wz]
        self.state   = state
        self.dt      = dt

        self.plFullState = np.empty((1,19))
    def __str__(self):
        return "payload m = {} kg, length of cable = {} m, \n\n Initial State = {}".format(self.mp, self.lc, self.state)

    def getPL_nextpos(self, fz, curr_posl, curr_vl, curr_p, curr_wl, curr_q):
        R_IB  = to_matrix(curr_q)
        pd    = np.cross(curr_wl, curr_p)
        u     = fz * R_IB * np.array([0,0,1]) 
        al    =  (1/self.mt) * (self.grav_ + (np.vdot(curr_p, R_IB @ np.array([0,0,fz])) - (self.m * self.lc * (np.vdot(pd, pd)))) * curr_p)
        Vl_   = al * self.dt + curr_vl
        posl_ = curr_vl * self.dt + curr_posl
        return posl_, Vl_

    def getPLAngularState(self, fz, curr_q, curr_p, curr_wl):
        R_IB = to_matrix(curr_q)
        wld  = (1/(self.lc*self.m)) * ( skew(-curr_p) @ R_IB @ np.array([0,0,fz]))
        wl_  = wld * self.dt + curr_wl
        pd    =  skew(curr_wl) @ curr_p
        p_    = pd*self.dt + curr_p
        return p_, wl_

    def PL_nextState(self, tau_inp, uav):
        curr_posl = self.state[0:3]   # position: x,y,z
        curr_vl   = self.state[3:6]   # linear velocity: xdot, ydot, zdot
        curr_p    = self.state[6:9]   # directional unit vector
        curr_wl   = self.state[9:12]  # Payload angular velocity in Inertial Frame
        curr_q    = self.state[12:16] # Quaternions: [qw, qx, qy, qz]
        curr_w    = self.state[16::]  # Quadrotor angular velocity

        fz       = tau_inp[0]
        tau_i    = tau_inp[1::]
        
        poslNext, VlNext  = self.getPL_nextpos(fz, curr_posl, curr_vl, curr_p, curr_wl, curr_q)
        pNext, wlNext     = self.getPLAngularState(fz, curr_q, curr_p, curr_wl)
        qNext, wNext      = uav.getNextAngularState(curr_w, curr_q, tau_i)
        
        
        self.state[0:3]   = poslNext   # position: x,y,z
        self.state[3:6]   = VlNext  # linear velocity: xdot, ydot, zdot
        self.state[6:9]   = pNext # directional unit vector
        self.state[9:12]  = wlNext # Payload angular velocity in Inertial Frame
        self.state[12:16] = qNext # Quadrotor attitude [q = qw, qx, qy, qz]
        self.state[16::]  = wNext # Quadrotor angular velocity [w = wx, wy, wz]
        self.plFullState  = np.vstack((self.plFullState, self.state))
        return self.state

    def cursorUp(self):
        ## This method removes the first row of the stack which is initialized as an empty array
        self.plFullState = np.delete(self.plFullState, 0, 0)

class SharedPayload:
    def __init__(self, payload_params, uavs_params):
        self.dt = payload_params['dt']
        self.g  = 9.81
        self.mp = float(payload_params['m_p']) # Mass of payload [kg]
        self.J = np.diag(payload_params['inertia'])
        self.mt_  = 0
        self.numOfquads = 0
        self.J_bar_term = np.zeros((3,3))
        if payload_params['payloadLead'] in 'enabled':
            self.lead = True
        else: 
            self.lead = False
        self.controller = payload_params['ctrlLee']
        self.cablegains = payload_params['cable_gains']
        self.posFrload = np.empty((1,3))
        for name, uav in uavs_params.items():
            self.posFrload = np.vstack((self.posFrload,np.array(uav['pos_fr_payload']).reshape((1,3))))
            self.mt_   += float(uav['m']) # Mass of quadrotors [kg] 
            self.J_bar_term += self.mt_ * skew(np.array(uav['pos_fr_payload']))@ skew(np.array(uav['pos_fr_payload']))
            self.numOfquads += 1
        self.J_bar = self.J - self.J_bar_term      
        self.mt    = self.mp + self.mt_ # total mass of quadrotors and payload
        self.grav_ = np.array([0,0,-self.mt*self.g])
        #state = [xp, yp, zp, xpd, ypd, zpd, qwp, qxp, qyp, qzp, wpx, wpy, wpz, q1,...,qn, w1,...,wn]
        self.plSysDim    = 6
        self.plStateSize = 13
        if np.linalg.det(self.J_bar) == 0:
            self.plSysDim -= 3
            self.plStateSize -= 7
            self.pointmass = True
            self.posFrload = np.delete(self.posFrload, 0, 0)

        self.sys_dim    = self.plSysDim + 3*self.numOfquads
        self.state_size = self.plStateSize + 6*self.numOfquads #13 for the payload and (3+3)*n for each cable angle and its derivative    
        self.plstate = np.empty((1,16+3*self.numOfquads))
        self.plFullState = np.empty((1,16+3*self.numOfquads))
        self.ctrlInp = np.empty((1,3))
        self.state = self.getInitState(uavs_params, payload_params)
        self.accl   = np.zeros(self.sys_dim,)
        
       
    
    def getInitState(self, uav_params, payload_params):
        self.state = np.zeros(self.state_size,)
        self.state[0:3]   = payload_params['init_pos_L']
        self.state[3:6]   = payload_params['init_linV_L']
        if not self.pointmass:
            init_ang     = np.radians(payload_params['init_angle']) 
            self.state[6:10]  = from_euler(init_ang[0], init_ang[1], init_ang[2])
            self.state[10:13] = payload_params['wl']
        j = self.plStateSize
        for initValues in uav_params.values():
            angR      = np.radians(initValues['q_dg'])
            self.state[j:j+3] = to_matrix(from_euler(angR[0], angR[1], angR[2], convention='xyz',axis_type='extrinsic')) @ np.array([0,0,-1])
            self.state[j+3*self.numOfquads:j+3+3*self.numOfquads] = initValues['qd']
            j+=3
        ctrlInp = np.empty((self.numOfquads,3))
        return self.state

    def getBq(self, uavs_params):
        Bq = np.zeros((self.sys_dim, self.sys_dim))
      
        Bq[0:3,0:3] = self.mt*np.identity(3)
      
        if not self.pointmass:
            Bq[3:6,3:6] = self.J_bar

        i = self.plSysDim
        k = self.plStateSize
      
        for name, uav in uavs_params.items():
            m = float(uav['m'])
            l = float(uav['l_c'])
            
            qi = self.state[k:k+3]
            k+=3

            if not self.pointmass:
                R_p = to_matrix(self.state[6:10])
                posFrload = uav['pos_fr_payload']
            
            Bq[i:i+3,0:3]    = -m*skew(qi) # Lee 2018
            Bq[i:i+3, i:i+3] = m*(l)*np.identity(3) # Lee 2018
            if not self.pointmass:
                Bq[0:3, 3:6]  += -m * R_p @ skew(np.array(posFrload))
                Bq[3:6,0:3]   +=  m * skew(np.array(posFrload)) @ np.transpose(R_p) 
                Bq[i:i+3, 3:6] = m*l*skew(qi) @ R_p @ skew(np.array(posFrload))
                Bq[3:6, i:i+3] = m*l*skew(np.array(posFrload)) @ np.transpose(R_p) @ skew(qi)
            
            i+=3
        return Bq

    def getNq(self, uavs_params):
        Nq =  np.zeros((self.sys_dim,))
        i = self.plSysDim
        k = self.plStateSize
        term = np.zeros((3,))
        Mq   = self.mt*np.identity(3)

        
       
        for name, uav in uavs_params.items():
            m = float(uav['m'])
            l = float(uav['l_c'])
           
            if not self.pointmass:
                posFrload = np.array(uav['pos_fr_payload'])
                R_p = to_matrix(self.state[6:10])
                wl = self.state[10:13]

            qi = self.state[k:k+3]
            wi = self.state[k+3*self.numOfquads:k+3+3*self.numOfquads]
            k+=3
            Nq[0:3]  -=  m*l*np.dot(wi,wi)*qi # Lee 2018
            Nq[i:i+3] = -m*skew(qi) @ np.array([0,0,-self.g]) # Lee 2018
            if not self.pointmass:
                Nq[0:3]  += m*R_p @ skew(wl) @skew(wl) @ posFrload
                Nq[3:6]  += m*l*skew(posFrload) @ np.transpose(R_p)*(np.linalg.norm(wi))**2 @ qi
                term     += skew(posFrload)@ np.transpose(R_p) @  np.array([0,0,-m*self.g])
                Nq[i:i+3] = m*l*skew(qi) @ R_p @ skew(wl) @skew(wl) @ posFrload  

            i+=3
        Nq[0:3] += Mq @ np.array([0,0,-self.g])
        if not self.pointmass:
            Nq[3:6] = -skew(wl)@self.J_bar@wl - Nq[3:6] + term
        
        return Nq


    def getuinp(self, uavs_params):        
        u_inp = np.zeros((self.sys_dim,))
        i, j, k = 0, self.plSysDim, self.plStateSize

        for name, uav in uavs_params.items():
            m = float(uav['m'])
            l = float(uav['l_c'])

            if not self.pointmass:
                R_p = to_matrix(self.state[6:10])
                wl = self.state[10:13]
                posFrload = np.array(uav['pos_fr_payload'])
          
            qi = self.state[k:k+3]
            wi = self.state[k+3*self.numOfquads:k+3+3*self.numOfquads]
            k+=3
            qiqiT = qi.reshape((3,1))@(qi.T).reshape((1,3))
            u_inp[0:3] += qiqiT @ self.ctrlInp[i,:]
            u_inp[j:j+3] = - skew(qi) @ (np.eye(3) - qiqiT) @  self.ctrlInp[i,:]

            if not self.pointmass:
                u_inp[3:6] += skew(posFrload)@np.transpose(R_p) @ self.ctrlInp[i,:]
                u_inp[j:j+3] += m * l * skew(qi) @ R_p @ skew(wl) @ skew(wl) @ posFrload
           
            i+=1
            j+=3
        return u_inp

    def getNextState(self, accl):
        # if not pointmass:
            #state = [xp, yp, zp, xpd, ypd, zpd, qwp, qxp, qyp, qzp, wpx, wpy, wpz, q1,...,qn, w1,..,wn]
        #else:
            #state = [xp, yp, zp, xpd, ypd, zpd, q1,...,qn, w1,...,wn]
        currVl  = np.zeros(self.sys_dim)
        currPos = np.zeros(self.sys_dim)

        currPos[0:3]  = self.state[0:3]
        currVl[0:3] = self.state[3:6]
        
        if not self.pointmass:
            currPos[3:7] = self.state[6:10]
            currVl[3:6] = self.state[10:13]     
       
        posNext = np.zeros_like(currPos)
        velNext = np.zeros_like(currVl)  
        velNext[0:3] = accl[0:3] * self.dt + currVl[0:3]
        posNext[0:3] = currVl[0:3] * self.dt + currPos[0:3]
        
        
        k = self.plStateSize        
        j = self.plSysDim
        for i in range(0, self.numOfquads):
            qi = self.state[k:k+3]
            wi = self.state[k+3*self.numOfquads:k+3+3*self.numOfquads]
            currVl[j:j+3] = wi
            wdi = accl[j:j+3]
            velNext[j:j+3] = wdi*self.dt + wi

            if not self.pointmass:
                currPos[j+1:j+4] = qi
                qd = np.cross(wi, qi)
                posNext[j+1:j+4] = qd*self.dt + qi  
            else:
                currPos[j:j+3] = qi
                qd = np.cross(wi, qi)
                posNext[j:j+3] = qd*self.dt + qi  
                qi = posNext[j:j+3]
            k+=3
            j+=3
        if not self.pointmass:
            posNext[3:7] = quat_integrate(currPos[3:7], currVl[3:6], self.dt)        
        return velNext, posNext

    def stateEvolution(self,torques, uavs, uavs_params):
        torques = np.delete(torques, 0,0)
    
        Bq    = self.getBq(uavs_params)
        Nq    = self.getNq(uavs_params)
        u_inp = self.getuinp(uavs_params)
        velNext, posNext = self.getNextState(self.accl)
        self.accl = np.linalg.inv(Bq)@(Nq + u_inp)
        self.state[0:3]   = posNext[0:3]
        self.state[3:6]   = velNext[0:3]
        if not self.pointmass:
            self.state[6:10]  = posNext[3:7]
            self.state[10:13] = velNext[3:6]
        k = self.plStateSize
        j = self.plSysDim
    
        self.plstate[0,0:3] = self.state[0:3]
        self.plstate[0,3:6] = self.state[3:6]
        for i in range(0, self.numOfquads):
            if not self.pointmass:
                self.state[k:k+3] = posNext[j+1:j+4]
                self.state[k+1+3*self.numOfquads:k+4+3*self.numOfquads] = velNext[j:j+3]
                self.plstate[0,k:k+3] = self.state[k:k+3]
                self.plstate[0,k+1+3*self.numOfquads:k+4+3*self.numOfquads] = velNext[j:j+3]

            else:
                self.state[k:k+3] = posNext[j:j+3]
                self.plstate[0,k:k+3] = self.state[k:k+3]
                self.state[k+3*self.numOfquads:k+3+3*self.numOfquads] = velNext[j:j+3]
                self.plstate[0,k+3*self.numOfquads:k+3+3*self.numOfquads] = velNext[j:j+3]

            k+=3
            j+=3
        m = 0
        for id in uavs.keys():
            tau    = torques[m,:].reshape(3,)
            curr_q = uavs[id].state[6:10]
            curr_w = uavs[id].state[10::]
            qNext, wNext = uavs[id].getNextAngularState(curr_w, curr_q, tau)
            uavs[id].state[6:10] = qNext
            uavs[id].state[10::] = wNext  
            m+=1
        return uavs, self.state 
    
    
    def stackCtrl(self, ctrlInp):  
       self.ctrlInp = np.vstack((self.ctrlInp,ctrlInp))

    def stackState(self):
        self.plFullState  = np.vstack((self.plFullState, self.plstate)) 
   
    def cursorPlUp(self):
        self.plFullState = np.delete(self.plFullState, 0, 0)

    def cursorUp(self):
        self.ctrlInp = np.delete(self.ctrlInp, 0, 0)
       

class UavModel:
    """initialize an instance of UAV object with the following physical parameters:
    m = 0.028 [kg]  -------------------------------------> Mass of the UAV
    I =   (16.571710 0.830806 0.718277
            0.830806 16.655602 1.800197    -----------------> Moment of Inertia 
            0.718277 1.800197 29.261652)*10^-6 [kg.m^2]"""

    def __init__(self, dt, state, uav_params, pload=False, lc=0):
        self.m        = float(uav_params['m'])
        self.I        = np.diag(uav_params['I'])
        self.invI     = linalg.inv(self.I)
        self.d        = float(uav_params['d']) 
        self.cft      = float(uav_params['cft'])
        self.all      = np.array([[1, 1, 1, 1],[0, -self.d, 0 , self.d],[self.d, 0 , -self.d, 0],[-self.cft, self.cft, -self.cft, self.cft]])
        self.invAll   = linalg.pinv(self.all)
        self.grav     = np.array([0,0,-self.m*9.81])
        self.pload    = pload # default is false (no payload)
        self.lc       = lc # default length of cable is zero (no payload)
            ### State initialized with the Initial values ###
            ### state = [x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, wx, wy, wz]
        self.state = state
        self.dt    = dt
        self.controller = uav_params['controller']
        self.fullState = np.empty((1,16))
        self.ctrlInps  = np.empty((1,8))
        if self.controller['name'] in 'lee':
            self.refState  = np.empty((1,12))
        else:
            self.refState  = np.empty((1,6))
        self.drag  = float((uav_params['drag']))
        if self.drag ==  1:
            self.Kaero = np.diag([-9.1785e-7, -9.1785e-7, -10.311e-7]) 
            
    def __str__(self):
        return "\nUAV object with physical parameters defined as follows: \n \n m = {} kg, l_arm = {} m \n \n{} {}\n I = {}{} [kg.m^2] \n {}{}\n\n Initial State = {}".format(self.m,self.d,'     ',self.I[0,:],' ',self.I[1,:],'     ',self.I[2,:], self.state)
        
    def getNextAngularState(self, curr_w, curr_q, tau):
        wdot  = self.invI @ (tau - skew(curr_w) @ self.I @ curr_w)
        wNext = wdot * self.dt + curr_w
        qNext = quat_integrate(curr_q, curr_w, self.dt)
        return qNext, wNext

    def getNextLinearState(self, curr_vel, curr_position, q ,fz, fa):
        R_IB = to_matrix(q)
        a =  (1/self.m) * (self.grav + R_IB @ np.array([0,0,fz]) + fa)
        velNext = a * self.dt + curr_vel
        posNext = curr_vel * self.dt + curr_position
        return posNext, velNext

    def states_evolution(self, tau_inp):
        """this method generates the 6D states evolution for the UAV given for each time step:
            the control input: f_th = [f1, f2, f3, f4] for the current step"""
        f_motors   = self.invAll @ tau_inp 
        w_motors   = self.wMotors(f_motors) #rotors angular velocities [rad/s]

        if self.drag == 1:
            fa = self.simpleDragModel(w_motors) # Simple Aerodynamic Drag Model
        else: 
            fa = np.zeros((3,))

        fz    = tau_inp[0]
        tau_i = tau_inp[1::]

        curr_pos  = self.state[0:3]  # position: x,y,z
        curr_vel  = self.state[3:6]  # linear velocity: xdot, ydot, zdot
        curr_q    = self.state[6:10] # quaternions: [qw, qx, qy, qz]
        curr_w    = self.state[10::]  # angular velocity: wx, wy, wz
        
        posNext, velNext = self.getNextLinearState(curr_vel, curr_pos, curr_q, fz, fa)
        qNext, wNext     = self.getNextAngularState(curr_w, curr_q, tau_i)
        
        self.state[0:3]  = posNext  # position: x,y,z
        self.state[3:6]  = velNext  # linear velocity: xdot, ydot, zdot
        self.state[6:10] = qNext# quaternions: [qw, qx, qy, qz]
        self.state[10::] = wNext # angular velocity: wx, wy, wz
    
        return self.state

    def stackStandCtrl(self, state, tau_inp, ref_state):
        ## This method stacks the actual and reference states of the UAV 
        ## and the control input vector [fz taux, tauy, tauz, f1, f2, f3, f4]
        curr_w = self.state[10::]
        wd    = self.invI @ (tau_inp[1::] - skew(curr_w) @ self.I @ curr_w)
        state = np.hstack((state,wd))
        self.fullState  = np.vstack((self.fullState, state))
        f_motors   = self.invAll @ tau_inp 
        self.ctrlInps   = np.vstack((self.ctrlInps, np.array([tau_inp, f_motors]).reshape(1,8)))
        self.refState   = np.vstack((self.refState, ref_state))
   
    def cursorUp(self):
        ## This method removes the first row of the stack which is initialized as an empty array
        self.fullState = np.delete(self.fullState, 0, 0)
        self.ctrlInps  = np.delete(self.ctrlInps,  0, 0)
        self.refState  = np.delete(self.refState,  0, 0)

    def wMotors(self, f_motor):
        """This method transforms the current thrust for each motor to command input to angular velocity  in [rad/s]"""
        coef    = [5.484560e-4, 1.032633e-6 , 2.130295e-11]
        w_motors = np.empty((4,))
        cmd = 0
        for i in range(0,len(f_motor)):
            coef[0] = coef[0] - f_motor[i]
            poly_   = poly(coef) 
            roots_  = poly_.roots()
            for j in range(0, 2):
                if (roots_[j] >= 0):
                    cmd = roots_[j]
            w_motors[i] = 0.04076521*cmd + 380.8359
        return w_motors    
    
    def simpleDragModel(self, w_motors):
        wSum = np.sum(w_motors)
        R_IB = to_matrix(self.state[6:10])
        fa   = wSum * self.Kaero @ np.transpose(R_IB) @ self.state[3:6]
        return fa



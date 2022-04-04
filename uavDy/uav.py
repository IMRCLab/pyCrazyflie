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
    def __init__(self, dt, state, uav_params):
        self.mp      = float(uav_params['m_p']) # Mass of payload [kg]
        self.lc      = float(uav_params['l_c'])  # length of cable [m]
        self.m       = float(uav_params['m']) # Mass of quadrotor [kg]
        self.mt      = self.m + self.mp # Total mass [kg]
        self.grav_   = np.array([0,0,-self.mt*9.81])
        # state = [xl, yl, zl, xldot, yldot, zldot, px, py, pz, wlx, wly, wlz, qw, qx, qy, qz, wx, wy, wz]
        self.state   = state
        self.dt      = dt

    def __str__(self):
        return "payload m = {} kg, length of cable = {} m, \n\n Initial State = {}".format(self.mp, self.lc, self.state)

    def getPL_nextpos(self, fz, curr_posl, curr_vl, curr_p, curr_wl, curr_q):
        R_IB  = to_matrix(curr_q)
        pd    = skew(curr_wl) @ curr_p
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
        
        pNext, wlNext     = self.getPLAngularState(fz, curr_q, curr_p, curr_wl)
        qNext, wNext      = uav.getNextAngularState(curr_w, curr_q, tau_i)
        poslNext, VlNext  = self.getPL_nextpos(fz, curr_posl, curr_vl, curr_p, curr_wl, curr_q)
        
        self.state[0:3]   = poslNext   # position: x,y,z
        self.state[3:6]   = VlNext  # linear velocity: xdot, ydot, zdot
        self.state[6:9]   = pNext # directional unit vector
        self.state[9:12]  = wlNext # Payload angular velocity in Inertial Frame
        self.state[12:16] = qNext # Quadrotor attitude [q = qw, qx, qy, qz]
        self.state[16::]  = wNext # Quadrotor angular velocity [w = wx, wy, wz]
        return self.state




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

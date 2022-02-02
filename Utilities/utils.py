import numpy as np
from rowan import to_matrix
def skew(w):
    w = w.reshape(3,1)
    w1 = w[0,0]
    w2 = w[1,0]
    w3 = w[2,0]
    return np.array([[0, -w3, w2],[w3, 0, -w1],[-w2, w1, 0]]).reshape((3,3))
    
def veeMap(w_tilde):
    w1 = w_tilde[2,1]
    w2 = w_tilde[0,2]
    w3 = w_tilde[1,0]
    return np.array([[w1],[w2],[w3]])

def Rx(angle):
    return np.array([[1, 0, 0],[0,np.cos(angle), np.sin(angle)],[0, -np.sin(angle), np.cos(angle)]])

def Ry(angle):
    return np.array([[np.cos(angle), 0, -np.sin(angle)],[0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])

def Rz(angle):
    return np.array([[np.cos(angle), np.sin(angle),0],[-np.sin(angle), np.cos(angle),0],[0, 0, 1]])

def RotatedCylinder(center_x, center_y, radius, height_z, q):
    R_i            = to_matrix(q) 
    z              = np.linspace(0, height_z, 50)
    theta          = np.linspace(0, 2*np.pi, 50)
    theta_grid, Zc = np.meshgrid(theta, z)
    Xc = radius*np.cos(theta_grid) + center_x
    Yc = radius*np.sin(theta_grid) + center_y
    Xb = np.zeros_like(Xc)
    Yb = np.zeros_like(Xc)
    Zb = np.zeros_like(Xc)
    for i in range(0,len(Xc.T)):
        for j in range(0,len(Xc.T)):
            rc = np.array([Xc[i,j],Yc[i,j],Zc[i,j]])
            rb = R_i @ rc
            Xb[i,j] = rb[0]
            Yb[i,j] = rb[1]
            Zb[i,j] = rb[2]   
    return Xb, Yb, Zb
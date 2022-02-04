import numpy as np
from rowan import from_matrix, to_matrix 
from utils import Rx, Ry, Rz

dt = 0.002
initPos = np.array([0,0,0.7])
initR = np.eye(3)
# initialize Rotation matrix about Roll-Pitch-Yaw
iangle = [0,0,0]
initR   = Rz((np.pi/180)*iangle[2]) @ Ry((np.pi/180)*iangle[1]) @ Rx((np.pi/180)*iangle[0])
initq = from_matrix(initR)
#Initialize Twist
initTwist = np.zeros((6,))
### State = [x, y, z, xdot, ydot, zdot, q1, q2, q3, q4, wx, wy, wz] ###
initState = np.zeros((13,))
initState[0:3]  = initPos  # position: x,y,z
initState[3:6]  = initTwist[0:3]  # linear velocity: xdot, ydot, zdot
initState[6:10] = initq# quaternions: [q1, q2, q3, q4]
initState[10::] = initTwist[3::] # angular velocity: wx, wy, wz

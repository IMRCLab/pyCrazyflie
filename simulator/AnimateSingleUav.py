import numpy as np
from utils import Rz, RotatedCylinder
from rowan import from_matrix, to_matrix 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import matplotlib.animation as animation
import uav



class PlotandAnimate:
    def __init__(self, fig, ax, uavModel,full_state, reference_state): 
        # Initialize the Actual and Reference states
        self.full_state      = full_state
        self.reference_state = reference_state
        self.uavModel        = uavModel
        # Initialize a 3d figure
        self.fig = fig#plt.figure(figsize=(10,10))
        self.ax  = ax#self.fig.add_subplot(autoscale_on=True,projection="3d")
        # self.ax.view_init(25, 35)
        # self.ax.view_init(25, -(90+35))
        self.ax.view_init(25,35)
        # Create the lines and vectors to draw body and desired frames
        self.line, = self.ax.plot(self.full_state[0,0:1], self.full_state[1,0:1], self.full_state[2,0:1], 'b--', lw=1)
        self.vec1  = self.ax.quiver([],[],[],[],[],[])
        self.vec2  = self.ax.quiver([],[],[],[],[],[])
        self.vec3  = self.ax.quiver([],[],[],[],[],[])
        self.vec1d = self.ax.quiver([],[],[],[],[],[])
        self.vec2d = self.ax.quiver([],[],[],[],[],[])
        self.vec3d = self.ax.quiver([],[],[],[],[],[])
        #Create the arms of the quadrotor in the body frame
        self.armb1  = np.array([[self.uavModel.d*10**(2)*np.cos(0)], [self.uavModel.d*10**(2)*np.sin(0)] ,[0]])
        self._armb1 = np.array([[-self.uavModel.d*10**(2)*np.cos(0)], [-self.uavModel.d*10**(2)*np.sin(0)] ,[0]])
        self.armb2  = Rz(np.pi/2) @ (self.armb1.reshape(3,))
        self._armb2 = Rz(np.pi/2) @ (self._armb1.reshape(3,))

    def startAnimation(self,videoname,show,save,dt):
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.full_state), interval=dt*1000,blit=True)
        if show:   
            plt.show()
        if save:
            self.ani.save('Videos/'+videoname)


    def plotFulltraj(self):
        self.setlimits()
        x = self.full_state[:,0]
        y = self.full_state[:,1]
        z = self.full_state[:,2]
        xref = self.reference_state[:,0]
        yref = self.reference_state[:,1]
        zref = self.reference_state[:,2]
        self.ax.plot3D(x, y, z, 'k--',lw=1.5)
        self.ax.plot3D(xref, yref ,zref,'g--',lw=1.5)
        ud = np.array([1,0,0])
        vd = np.array([0,1,0])  
        wd = np.array([0,0,1])
        for i in range(0,len(x)):
            q = self.full_state[i,6:10]
            R_i = to_matrix(q)
            u = R_i[:,0]
            v = R_i[:,1]
            w = R_i[:,2]
            self.vec1  = self.ax.quiver(x[i],y[i],z[i], u[0], u[1] ,u[2],color = 'r', length = 0.2)
            self.vec2  = self.ax.quiver(x[i],y[i],z[i], v[0], v[1] ,v[2],color = 'g', length = 0.2)  
            self.vec3  = self.ax.quiver(x[i],y[i],z[i], w[0], w[1] ,w[2],color = 'b', length = 0.2)
           
            self.vec1  = self.ax.quiver(x,y,z, u[0], u[1] ,u[2],color = 'r', length = 0.2)
            self.vec2  = self.ax.quiver(x,y,z, v[0], v[1] ,v[2],color = 'g', length = 0.2)
            self.vec3  = self.ax.quiver(x,y,z, w[0], w[1] ,w[2],color = 'b', length = 0.2)
            self.vec1r = self.ax.quiver(xref,yref,zref, ud[0], ud[1] ,ud[2],color = 'r', length = 0.3)
            self.vec2r = self.ax.quiver(xref,yref,zref, vd[0], vd[1] ,vd[2],color = 'g', length = 0.3)
            self.vec3r = self.ax.quiver(xref,yref,zref, wd[0], wd[1] ,wd[2],color = 'b', length = 0.3)
        self.ax.plot3D(x, y, z, 'k--',lw=1.5)    
        plt.show()

    def setlimits(self):
        edge  = 0.9
        max_x = max(self.full_state[0,:])
        max_y = max(self.full_state[1,:])
        max_z = max(self.full_state[2,:])
       
        self.ax.set_xlim3d([-max_x-edge, max_x+edge])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-max_y-edge, max_y+edge])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-max_z-edge, max_z+edge]) #
        self.ax.set_zlabel('Z')

    def drawQuivers(self, x, y, z, q, xref, yref, zref):
        R_i = to_matrix(q)
        u = R_i[:,0]
        v = R_i[:,1]
        w = R_i[:,2]
        ud = np.array([1,0,0])
        vd = np.array([0,1,0])
        wd = np.array([0,0,1])
        self.vec1  = self.ax.quiver(x,y,z, u[0], u[1] ,u[2],color = 'r', length = 0.2)
        self.vec2  = self.ax.quiver(x,y,z, v[0], v[1] ,v[2],color = 'g', length = 0.2)
        self.vec3  = self.ax.quiver(x,y,z, w[0], w[1] ,w[2],color = 'b', length = 0.2)
        self.vec1r = self.ax.quiver(xref,yref,zref, ud[0], ud[1] ,ud[2],color = 'r', length = 0.5)
        self.vec2r = self.ax.quiver(xref,yref,zref, vd[0], vd[1] ,vd[2],color = 'g', length = 0.5)
        self.vec3r = self.ax.quiver(xref,yref,zref, wd[0], wd[1] ,wd[2],color = 'b', length = 0.5)
        

    def getCurrState(self, i):
        x = self.full_state[:i+1,0]
        y = self.full_state[:i+1,1]
        z = self.full_state[:i+1,2]
        q = self.full_state[i,6:10].reshape(4,)
        return x, y, z, q

    def getRefState(self, i):
        xref = self.reference_state[:i+1,0]
        yref = self.reference_state[:i+1,1]
        zref = self.reference_state[:i+1,2]
        return xref, yref, zref

    def getArmpos(self, x, y, z, q):
        R_i      = to_matrix(q)
        position = np.array([x, y, z]) 
        armI1    = position + R_i @ (self.armb1.reshape(3,))
        _armI1   = position + R_i @ (self._armb1.reshape(3,))
        armI2    = position + R_i @(self.armb2.reshape(3,))
        _armI2   = position + R_i @ (self._armb2.reshape(3,))
        return armI1, armI2, _armI1, _armI2

    def drawActvsRefTraj(self, x, y, z, xref, yref, zref):
            self.ax.plot3D(x, y, z, 'k-.',lw=1.5,label="Actual Trajectory")
            self.ax.plot3D(xref, yref ,zref,'g--',lw=1.5,label="Reference Trajectory")
            self.ax.legend()

    def drawQuadrotorArms(self, x, y, z, armI1, armI2, _armI1, _armI2):
        self.ax.plot3D(np.linspace(x, armI1[0]), np.linspace(y, armI1[1]), np.linspace(z, armI1[2]),'k',lw=2)
        self.ax.plot3D(np.linspace(x, _armI1[0]), np.linspace(y, _armI1[1]), np.linspace(z, _armI1[2]),'k',lw=2)
        
        self.ax.plot3D(np.linspace(x, armI2[0]), np.linspace(y, armI2[1]), np.linspace(z, armI2[2]),'k',lw=2)
        self.ax.plot3D(np.linspace(x, _armI2[0]), np.linspace(y, _armI2[1]), np.linspace(z, _armI2[2]),'k',lw=2)

    def drawPropellers(self, Xb, Yb, Zb,armI1, armI2, _armI1, _armI2):
        self.ax.plot_surface(Xb+armI1[0], Yb+armI1[1], Zb+armI1[2], alpha=1)
        self.ax.plot_surface(Xb+_armI1[0], Yb+_armI1[1], Zb+_armI1[2], alpha=1)
        self.ax.plot_surface(Xb+armI2[0], Yb+armI2[1], Zb+armI2[2], alpha=1)
        self.ax.plot_surface(Xb+_armI2[0], Yb+_armI2[1], Zb+_armI2[2], alpha=1)

    def animate(self,i):
        self.ax.cla()
        self.setlimits()
        x, y, z, q                   = self.getCurrState(i)
        xref,yref,zref               = self.getRefState(i) 
        armI1, armI2, _armI1, _armI2 = self.getArmpos(x[i],y[i],z[i],q)

        self.drawQuivers(x[i],y[i],z[i], q, xref[i], yref[i], zref[i])
        self.drawActvsRefTraj(x, y, z, xref, yref, zref)
        self.drawQuadrotorArms(x[i], y[i], z[i], armI1, armI2, _armI1, _armI2)

        Xb,Yb,Zb = RotatedCylinder(0,0,0.1,0.1,q) 
        self.drawPropellers(Xb, Yb, Zb,armI1, armI2, _armI1, _armI2)

        return self.line, 



import numpy as np
import rowan as rn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import SubplotSpec
from uavDy import uav

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    row.set_title('\n\n\n'+title, fontweight='medium',fontsize='small')
    row.set_frame_on(False)
    row.axis('off')

def outputPlots(ref_state, full_state, showPlot, tf_sim, pdfName):
    print('Plotting...')
    plt.rcParams['axes.grid'] = True
    
    fig1, ax1 = plt.subplots(2, 3, sharex=True ,sharey=True, squeeze=True)
    # fig1.set_size_inches(18, 4)
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots(2, 3, sharex=True, sharey=True, squeeze=True)
    # fig2.set_size_inches(18, 4)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(3, 1, sharex=True ,sharey=True, squeeze=True)
    # fig3.set_size_inches(18, 4)
    fig3.tight_layout()

    time   = np.linspace(0, tf_sim*1e-3, num=len(full_state)) 
    pos    = full_state[:,0:3]
    linVel = full_state[:,3:6]
    angVel = full_state[:,10::]
  
    posdes    = ref_state[:,0:3]
    linVeldes = ref_state[:,3::]
    ts = 'time [s]'
    
    ax1[0,0].plot(time, pos[:,0]), ax1[0,1].plot(time, pos[:,1]), ax1[0,2].plot(time, pos[:,2])
    ax1[0,0].set_ylabel('x [m]',), ax1[0,1].set_ylabel('y [m]'), ax1[0,2].set_ylabel('z [m]')
    ax1[1,0].plot(time, posdes[:,0]), ax1[1,1].plot(time, posdes[:,1]), ax1[1,2].plot(time, posdes[:,2])
    ax1[1,0].set_ylabel('x des [m]'), ax1[1,1].set_ylabel('y des [m]'), ax1[1,2].set_ylabel('z des [m]')
    fig1.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(2,3)
    create_subtitle(fig1, grid[0, ::], 'Actual Positions')
    create_subtitle(fig1, grid[1, ::], 'Reference Positions')
      
    ax2[0,0].plot(time, linVel[:,0]), ax2[0,1].plot(time, linVel[:,1]), ax2[0,2].plot(time, linVel[:,2])
    ax2[0,0].set_ylabel('vx [m/s]'), ax2[0,1].set_ylabel('vy [m/s]'), ax2[0,2].set_ylabel('vz [m/s]')
    
    ax2[1,0].plot(time, linVeldes[:,0]), ax2[1,1].plot(time, linVeldes[:,1]), ax2[1,2].plot(time, linVeldes[:,2])
    ax2[1,0].set_ylabel('vx des [m/s]'), ax2[1,1].set_ylabel('vy des [m/s]'), ax2[1,2].set_ylabel('vz des [m/s]')
    fig2.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(2,3)
    create_subtitle(fig2, grid[0, ::], 'Actual Linear Velocities')
    create_subtitle(fig2, grid[1, ::], 'Reference Linear Velocities')
    
    ax3[0].plot(time, angVel[:,0])
    ax3[1].plot(time, angVel[:,1])
    ax3[2].plot(time, angVel[:,2])
    ax3[0].set_ylabel('wx [deg/s]',labelpad=-5), ax3[1].set_ylabel('wy [deg/s]',labelpad=-5), ax3[2].set_ylabel('wz [deg/s]',labelpad=-5)
    fig3.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig3, grid[0, ::], 'Actual Angular Velocities')
    
    with PdfPages(pdfName) as pdf:
      fig1.savefig(pdf, format='pdf', bbox_inches='tight')
      fig2.savefig(pdf, format='pdf', bbox_inches='tight')
      fig3.savefig(pdf, format='pdf', bbox_inches='tight')
    if showPlot:
        plt.show()


def RotatedCylinder(center_x, center_y, radius, height_z, q):
    R_i            = rn.to_matrix(q) 
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
        q90z        = rn.from_euler(0, 0, np.radians(90),convention='xyz')
        rot90z      = rn.to_matrix(q90z)
        self.armb2  = rot90z @ (self.armb1.reshape(3,))
        self._armb2 = rot90z @ (self._armb1.reshape(3,))

    def startAnimation(self,videoname,dt):
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.full_state), interval=dt*1000,blit=True)
        self.ani.save('Videos/'+videoname)

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
        R_i = rn.to_matrix(q)
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
        R_i      = rn.to_matrix(q)
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



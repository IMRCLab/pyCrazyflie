import numpy as np
import rowan as rn
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits import mplot3d 
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import SubplotSpec, GridSpec
from uavDy import uav

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    row.set_title('\n\n\n'+title, fontweight='medium',fontsize='medium')
    row.set_frame_on(False)
    row.axis('off')

def setlimits(ax, full_state):
    # This method finds the maximum value in the x-y-z actual states and sets the limits of the figure accordingly   
    # edge: adds extra space for the figure 
    edge  = 0.9
    max_x = max(full_state[:,0])
    max_y = max(full_state[:,1])
    max_z = max(full_state[:,2])
    if (max_x >= max_y) and (max_x >= max_z):
        max_ = max_x
        ax.set_xlim3d([-max_-edge, max_+edge])
        ax.set_ylim3d([-max_-edge, max_+edge])
        ax.set_zlim3d([-max_-edge, max_+edge])
    elif (max_y >= max_x) and (max_y >= max_z):
        max_ = max_y
        ax.set_xlim3d([-max_-edge, max_+edge])
        ax.set_ylim3d([-max_-edge, max_+edge])
        ax.set_zlim3d([-max_-edge, max_+edge])
    else:
        max_ = max_z
        ax.set_xlim3d([-max_-edge, max_+edge])
        ax.set_ylim3d([-max_-edge, max_+edge])
        ax.set_zlim3d([-max_-edge, max_+edge])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax

def plotPayloadStates(full_state, tf_sim):
    """This function plots the states of the payload"""
    # PL_states = [xl, vl, p, wl]
    fig8, ax11 = plt.subplots(3, 1, sharex=True ,sharey=True)
    fig8.tight_layout()
    
    fig9, ax12 = plt.subplots(3, 1, sharex=True, sharey=True)
    fig9.tight_layout()

    fig10, ax13 = plt.subplots(3, 1, sharex=True ,sharey=True)
    fig10.tight_layout()

    fig11, ax14 = plt.subplots(3, 1, sharex=True ,sharey=True)
    fig11.tight_layout()

    time   = np.linspace(0, tf_sim*1e-3, num=len(full_state)) 
    pos    = full_state[:,0:3]
    linVel = full_state[:,3:6]
    angVel = full_state[:,9:12]
    p      = full_state[:,6:9]
    ts = 'time [s]'
###############################################################################################
   
    ax11[0].plot(time, pos[:,0], c='k', lw=0.75, label='Actual'), ax11[1].plot(time, pos[:,1], lw=0.75, c='k'), ax11[2].plot(time, pos[:,2], lw=0.75, c='k')
    ax11[0].set_ylabel('x [m]',), ax11[1].set_ylabel('y [m]'), ax11[2].set_ylabel('z [m]')
    ax11[0].legend()
    fig8.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig8, grid[0, ::], 'Actual Payload Positions')
###############################################################################################

   
    ax12[0].plot(time, linVel[:,0],lw=0.75, c='k', label='Actual'), ax12[1].plot(time, linVel[:,1],lw=0.75, c='k'), ax12[2].plot(time, linVel[:,2],lw=0.75, c='k')
    ax12[0].set_ylabel('vx [m/s]'), ax12[1].set_ylabel('vy [m/s]'), ax12[2].set_ylabel('vz [m/s]')
    ax12[0].legend()
    fig9.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig9, grid[0, ::], 'Actual Payload Linear Velocities')

###############################################################################################

    ax13[0].plot(time, angVel[:,0],c='k',lw=1, label='Actual'),  ax13[1].plot(time, angVel[:,1],c='k',lw=1), ax13[2].plot(time, angVel[:,2],c='k',lw=1)
    ax13[0].set_ylabel('wx [deg/s]',labelpad=-5), ax13[1].set_ylabel('wy [deg/s]',labelpad=-5), ax13[2].set_ylabel('wz [deg/s]',labelpad=-5)
    fig10.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig10, grid[0, ::], ' Actual Payload Angular Velocities')

###############################################################################################
    
    ax14[0].plot(time, p[:,0],c='k',lw=1, label='Actual'), ax14[1].plot(time, p[:,1],c='k',lw=1), ax14[2].plot(time, p[:,2],c='k',lw=1)
    ax14[0].set_ylabel('px',labelpad=-5), ax14[1].set_ylabel('py',labelpad=-5), ax14[2].set_ylabel('pz',labelpad=-5)
    fig11.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig11, grid[0, ::], 'Cable Directional Unit Vector')

    return fig8, fig9, fig10, fig11



###############################################################################################
def outputPlots(uavModel, plFullstate, ref_state, full_state, cont_stack, savePlot, tf_sim, pdfName):
    print('Plotting...')
    plt.rcParams['axes.grid'] = True
    
    fig1, ax1 = plt.subplots(3, 1, sharex=True ,sharey=True)
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots(3, 1, sharex=True, sharey=True)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(3, 1, sharex=True ,sharey=True)
    fig3.tight_layout()

    fig4, ax4 = plt.subplots(2, 3, sharex=True ,sharey=True)
    fig4.tight_layout()

    fig5 = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig5)

    ax5 = fig5.add_subplot(gs[:, 0])
    ax6 = fig5.add_subplot(gs[0,1])
    ax7 = fig5.add_subplot(gs[1,1],sharey=ax6)
    ax8 = fig5.add_subplot(gs[2,1],sharey=ax6)

    fig6, ax9 = plt.subplots(4, 1, sharex=True ,sharey=True,figsize=(9,4.8))
    fig6.tight_layout()


    time   = np.linspace(0, tf_sim*1e-3, num=len(full_state)) 
    pos    = full_state[:,0:3]
    linVel = full_state[:,3:6]
    angVel = full_state[:,10::]
  
    posdes    = ref_state[:,0:3]
    linVeldes = ref_state[:,3::]
    ts = 'time [s]'
    
    poserr  = (posdes[:,:] - pos[:,:]).reshape(len(full_state),3)
    linVerr = (linVeldes[:,:] - linVel[:,:]).reshape(len(full_state),3) 

    ###################################

    ax1[0].plot(time, pos[:,0], c='k', lw=0.75,label='Actual'), ax1[1].plot(time, pos[:,1], lw=0.75, c='k'), ax1[2].plot(time, pos[:,2], lw=0.75, c='k')
    ax1[0].plot(time, posdes[:,0], lw=0.75, c='darkgreen',label='Reference'), ax1[1].plot(time, posdes[:,1], lw=0.75, c='darkgreen'), ax1[2].plot(time, posdes[:,2], lw=0.75, c='darkgreen')
    ax1[0].set_ylabel('x [m]',), ax1[1].set_ylabel('y [m]'), ax1[2].set_ylabel('z [m]')
    ax1[0].legend()
    fig1.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig1, grid[0, ::], 'Actual vs Reference Positions')

    ###################################
      
    ax2[0].plot(time, linVel[:,0],lw=0.75, c='k' ,label='Actual'), ax2[1].plot(time, linVel[:,1],lw=0.75, c='k'), ax2[2].plot(time, linVel[:,2],lw=0.75, c='k')
    ax2[0].plot(time, linVeldes[:,0],lw=0.75, c='darkgreen',label='Reference'), ax2[1].plot(time, linVeldes[:,1],lw=0.75, c='darkgreen'), ax2[2].plot(time, linVeldes[:,2],lw=0.75, c='darkgreen')
    ax2[0].set_ylabel('vx [m/s]'), ax2[1].set_ylabel('vy [m/s]'), ax2[2].set_ylabel('vz [m/s]')
    ax2[0].legend()
    fig2.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig2, grid[0, ::], 'Actual vs Reference Linear Velocities')

    ###################################

    ax3[0].plot(time, angVel[:,0],c='k',lw=1)
    ax3[1].plot(time, angVel[:,1],c='k',lw=1)
    ax3[2].plot(time, angVel[:,2],c='k',lw=1)
    ax3[0].set_ylabel('wx [deg/s]',labelpad=-5), ax3[1].set_ylabel('wy [deg/s]',labelpad=-5), ax3[2].set_ylabel('wz [deg/s]',labelpad=-5)
    fig3.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig3, grid[0, ::], 'Actual Angular Velocities')

    ###################################

    ax4[0,0].plot(time, poserr[:,0],c='r',lw=0.7), ax4[0,1].plot(time, poserr[:,1],c='r',lw=0.7), ax4[0,2].plot(time, poserr[:,2],c='r',lw=0.7)
    ax4[0,0].set_ylabel('ex [m/s]'), ax4[0,1].set_ylabel('ey [m/s]'), ax4[0,2].set_ylabel('ez [m/s]')
    
    ax4[1,0].plot(time, linVerr[:,0],c='r',lw=0.7), ax4[1,1].plot(time, linVerr[:,1],c='r',lw=0.7), ax4[1,2].plot(time, linVerr[:,2],c='r',lw=0.7)
    ax4[1,0].set_ylabel('vex des [m/s]'), ax4[1,1].set_ylabel('vey des [m/s]'), ax4[1,2].set_ylabel('vez des [m/s]')
    fig4.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(2,3)
    create_subtitle(fig4, grid[0, ::], 'Positional errors')
    create_subtitle(fig4, grid[1, ::], 'Linear Velocities errors')
    
    ###################################

    ax5.plot(time, cont_stack[:,0],lw=0.8, c='darkblue')
    ax5.set_ylabel('fz [N]')
    
    ax6.plot(time, cont_stack[:,1], lw=0.8, c='darkblue'), ax7.plot(time, cont_stack[:,2], lw=0.8, c='darkblue'), ax8.plot(time, cont_stack[:,3],lw=0.8, c='darkblue')
    ax6.set_ylabel('taux [N.m]',fontsize='small'), ax7.set_ylabel('tauy [N.m]',fontsize='small'), ax8.set_ylabel('tauz [N.m]',fontsize='small')
    fig5.supxlabel(ts,fontsize='small')

    create_subtitle(fig5, gs[::, 0], 'Force Control Input')
    create_subtitle(fig5, gs[::, 1], 'Torque Control Input')

    ###################################
    ax9[0].plot(time, cont_stack[:,4], c='darkred',lw=0.7)
    ax9[1].plot(time, cont_stack[:,5], c='darkred',lw=0.7)
    ax9[2].plot(time, cont_stack[:,6], c='darkred',lw=0.7)
    ax9[3].plot(time, cont_stack[:,7], c='darkred',lw=0.7)
    ax9[0].set_ylabel('f1 [N]'), ax9[1].set_ylabel('f2 [N]'), ax9[2].set_ylabel('f3 [N]'), ax9[3].set_ylabel('f4 [N]')
    fig6.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(4,1)
    create_subtitle(fig6, grid[0,::], 'Motor Forces')
    ###################################


    fig7 = plt.figure(figsize=(10,10))
    ax10 = fig7.add_subplot(autoscale_on=True,projection="3d")
    ax10.plot3D(pos[:,0], pos[:,1], pos[:,2], 'k-.',lw=1.5, label="Actual Trajectory")
    ax10.plot3D(posdes[:,0], posdes[:,1] , posdes[:,2],'darkgreen',ls='--',lw=1.5,label="Reference Trajectory")
    ax10.legend()
    ax10 = setlimits(ax10, pos)

    if uavModel.pload :
        fig8, fig9, fig10, fig11 = plotPayloadStates(plFullstate, tf_sim)
    if savePlot:
        with PdfPages(pdfName) as pdf:
            fig1.savefig(pdf, format='pdf', bbox_inches='tight')
            fig2.savefig(pdf, format='pdf', bbox_inches='tight')
            fig3.savefig(pdf, format='pdf', bbox_inches='tight')
            fig4.savefig(pdf, format='pdf', bbox_inches='tight')  
            fig5.savefig(pdf, format='pdf', bbox_inches='tight')  
            fig6.savefig(pdf, format='pdf', bbox_inches='tight')
            fig7.savefig(pdf, format='pdf', bbox_inches='tight')
            fig8.savefig(pdf, format='pdf', bbox_inches='tight')
            fig9.savefig(pdf, format='pdf', bbox_inches='tight')
            fig10.savefig(pdf, format='pdf', bbox_inches='tight')
            fig11.savefig(pdf, format='pdf', bbox_inches='tight')
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

def Sphere(Cx, Cy, Cz, r):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = r*np.cos(u) * np.sin(v)
    y = r*np.sin(u) * np.sin(v)
    z = r*np.cos(v)
    return x, y, z
class PlotandAnimate:
    def __init__(self, fig, ax, uavModel, plFullstate, full_state, reference_state): 
        # Initialize the Actual and Reference states
        self.full_state      = full_state
        self.reference_state = reference_state
        self.plFullstate     = plFullstate
        self.uavModel        = uavModel
        # Initialize a 3d figure
        self.fig = fig
        self.ax  = ax
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
        # This method finds the maximum value in the x-y-z actual states and sets the limits of the figure accordingly   
        # edge: adds extra space for the figure 
        edge  = 0.5
        max_x = max(self.full_state[:,0])
        max_y = max(self.full_state[:,1])
        max_z = max(self.full_state[:,2])
        if (max_x >= max_y) and (max_x >= max_z):
            max_ = max_x
            self.ax.set_xlim3d([-max_-edge, max_+edge])
            self.ax.set_ylim3d([-max_-edge, max_+edge])
            self.ax.set_zlim3d([-max_-edge, max_+edge])
        elif (max_y >= max_x) and (max_y >= max_z):
            max_ = max_y
            self.ax.set_xlim3d([-max_-edge, max_+edge])
            self.ax.set_ylim3d([-max_-edge, max_+edge])
            self.ax.set_zlim3d([-max_-edge, max_+edge])
        else:
            max_ = max_z
            self.ax.set_xlim3d([-max_-edge, max_+edge])
            self.ax.set_ylim3d([-max_-edge, max_+edge])
            self.ax.set_zlim3d([-max_-edge, max_+edge])
       
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
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
            self.ax.plot3D(xref, yref ,zref,c='darkgreen',ls='--',lw=1.5,label="Reference Trajectory")
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

    def getPayloadStates(self,i):
        xl = self.plFullstate[:i+1,0]
        yl = self.plFullstate[:i+1,1]
        zl = self.plFullstate[:i+1,2]
        return xl, yl, zl

    def drawPlTraj(self, xl,yl,zl):
        self.ax.plot3D(xl, yl, zl, 'darkblue',linestyle='-.',lw=1.5,label="Payload Trajectory")
        self.ax.legend()

    def drawPayload(self,x,y,z,xl,yl,zl):
        c_st = np.array([x,y,z])
        c_en = np.array([xl,yl,zl])
        self.ax.plot3D(np.linspace(c_st[0], c_en[0]), np.linspace(c_st[1], c_en[1]), np.linspace(c_st[2], c_en[2]), 'darkblue',lw=2)

    def animate(self,i):
        self.ax.cla()
        self.setlimits()
        x, y, z, q                   = self.getCurrState(i)
        xref,yref,zref               = self.getRefState(i) 
        armI1, armI2, _armI1, _armI2 = self.getArmpos(x[i],y[i],z[i],q)

        if self.uavModel.pload:
            xl, yl, zl  = self.getPayloadStates(i)
            self.drawPayload(x[i], y[i], z[i], xl[i], yl[i], zl[i])
            self.drawPlTraj(xl, yl, zl)
            r = 1e-1
            xsp, ysp, zsp = Sphere(xl, yl, zl, r)
            self.ax.plot_surface(xl[i]+xsp, yl[i]+ysp, zl[i]+zsp, cmap=plt.cm.YlGnBu_r)
        self.drawQuivers(x[i],y[i],z[i], q, xref[i], yref[i], zref[i])
        self.drawActvsRefTraj(x, y, z, xref, yref, zref)
        self.drawQuadrotorArms(x[i], y[i], z[i], armI1, armI2, _armI1, _armI2)

        Xb,Yb,Zb = RotatedCylinder(0,0,0.1,0.1,q) 
        self.drawPropellers(Xb, Yb, Zb,armI1, armI2, _armI1, _armI2)

        return self.line, 



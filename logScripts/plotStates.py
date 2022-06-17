import cfusdlog
import matplotlib.pyplot as plt
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits import mplot3d 
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import SubplotSpec, GridSpec

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    row.set_title('\n\n\n'+title, fontweight='medium',fontsize='medium')
    row.set_frame_on(False)
    row.axis('off')


def setlimits(ax, full_state):
    # This method finds the maximum value in the x-y-z actual states and sets the limits of the figure accordingly   
    # edge: adds extra space for the figure 
    edge  = 0.4
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

def preparefigs():
    fig1, ax1 = plt.subplots(3, 1, sharex=True ,sharey=True)
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots(3, 1, sharex=True, sharey=True)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(3, 1, sharex=True ,sharey=True)
    fig3.tight_layout()

    fig4, ax4 = plt.subplots(3, 1, sharex=True ,sharey=True)
    fig4.tight_layout()

    fig5, ax5 = plt.subplots(2, 3, sharex=True ,sharey=True)
    fig5.tight_layout()

    fig6 = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig6)
    ax6 = fig6.add_subplot(gs[:, 0])
    ax7 = fig6.add_subplot(gs[0,1])
    ax8 = fig6.add_subplot(gs[1,1],sharey=ax7)
    ax9 = fig6.add_subplot(gs[2,1],sharey=ax7)
    
    fig7 = plt.figure(figsize=(10,10))
    ax10 = fig7.add_subplot(autoscale_on=True,projection="3d")

    return fig1, ax1, fig2, ax2, fig3, ax3, fig4, ax4, fig5, ax5, fig6, gs, ax6, ax7, ax8, ax9, fig7, ax10

def main(args):
    # decode binary log data
    logData = cfusdlog.decode(args.file_usd)['fixedFrequency']
 
    logDataKeys = list(logData.keys())
    controller = args.controller

    if controller in 'sjc':
        ctrl = 'ctrlSJC.'
    elif controller in 'lee':
        ctrl = 'ctrlLee.'
        Kpxy = logData['ctrlLee.Kpos_Px'][0]
        Kpz  = logData['ctrlLee.Kpos_Pz'][0]
        Kdxy = logData['ctrlLee.Kpos_Dx'][0]
        Kdz  = logData['ctrlLee.Kpos_Dz'][0]

        Krxy = logData['ctrlLee.KR_x'][0]
        Krz  = logData['ctrlLee.KR_z'][0]
        Kwxy = logData['ctrlLee.Kw_x'][0]
        Kwz  = logData['ctrlLee.Kw_z'][0]             
                     
    filename = args.filename
    print('Plotting...')
    f = PdfPages(filename)
    
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.max_open_warning'] = 100

    pos = np.column_stack((
    logData['stateEstimateZ.x'] / 1000.0,
    logData['stateEstimateZ.y'] / 1000.0,
    logData['stateEstimateZ.z'] / 1000.0))

    linVel = np.column_stack((
    logData['stateEstimateZ.vx'] / 1000.0,
    logData['stateEstimateZ.vy'] / 1000.0,
    logData['stateEstimateZ.vz'] / 1000.0))

    
    posdes = np.column_stack((
    logData['ctrltargetZ.x'] / 1000.0,
    logData['ctrltargetZ.y'] / 1000.0,
    logData['ctrltargetZ.z'] / 1000.0))

    linVeldes = np.column_stack((
    logData['ctrltargetZ.vx'] / 1000.0,
    logData['ctrltargetZ.vy'] / 1000.0,
    logData['ctrltargetZ.vz'] / 1000.0))

    angVel = np.column_stack((
    logData[ctrl+'omegax'],
    logData[ctrl+'omegay'],
    logData[ctrl+'omegaz']))

    angVeldes = np.column_stack((
    logData[ctrl+'omegarx'],
    logData[ctrl+'omegary'],
    logData[ctrl+'omegarz']))

    cont_stack = np.column_stack((
    logData[ctrl+'thrustSI'],
    logData[ctrl+'torquex'] ,
    logData[ctrl+'torquey'] ,
    logData[ctrl+'torquez'] ))
 
    if controller in 'lee':
        rpydes = np.column_stack((
        logData['ctrlLee.rpydx'],
        logData['ctrlLee.rpydy'],
        logData['ctrlLee.rpydz']))
        
        rpy = np.column_stack((
        logData['ctrlLee.rpyx'],
        logData['ctrlLee.rpyy'],
        logData['ctrlLee.rpyz']))
        
    elif controller in 'sjc':
        rpydes = np.column_stack((
        logData['ctrlSJC.qx'],
        logData['ctrlSJC.qy'],
        logData['ctrlSJC.qz']))

        rpy = np.column_stack((
        logData['ctrlSJC.qrx'],
        logData['ctrlSJC.qry'],
        logData['ctrlSJC.qrz']))
    
        

    poserr = (posdes[:,:] - pos[:,:]).reshape(len(pos),3)
    linVerr = (linVeldes[:,:] - linVel[:,:]).reshape(len(pos),3) 


    ts = 'time [s]'
    time = np.column_stack(logData['timestamp']/1000).reshape(len(pos),)
    time = time - time[0]
    fig1, ax1, fig2, ax2, fig3, ax3, fig4, ax4, fig5, ax5, \
    fig6, gs, ax6, ax7, ax8, ax9, fig7 , ax10\
     = preparefigs()
    

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
    ax3[0].plot(time, np.degrees(rpy[:,0]), c='k',lw=0.75,label='Actual')
    ax3[1].plot(time, np.degrees(rpy[:,1]), c='k',lw=0.75,label='Actual')
    ax3[2].plot(time, np.degrees(rpy[:,2]), c='k',lw=0.75,label='Actual')
    ax3[0].plot(time, np.degrees(rpydes[:,0]) ,lw=0.75, c='darkgreen',label='Reference')
    ax3[1].plot(time, np.degrees(rpydes[:,1]) ,lw=0.75, c='darkgreen',label='Reference')
    ax3[2].plot(time, np.degrees(rpydes[:,2]) ,lw=0.75, c='darkgreen',label='Reference')

    ax3[0].set_ylabel('r [deg]',labelpad=-2), ax3[1].set_ylabel('p [deg]',labelpad=-2), ax3[2].set_ylabel('y [deg]',labelpad=-2)
    fig3.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig3, grid[0, ::], 'Actual vs Reference Angular Velocities')

    ###################################
    ax4[0].plot(time, np.degrees(angVel[:,0]), c='k',lw=0.75,label='Actual')
    ax4[1].plot(time, np.degrees(angVel[:,1]), c='k',lw=0.75,label='Actual')
    ax4[2].plot(time, np.degrees(angVel[:,2]), c='k',lw=0.75,label='Actual')
    ax4[0].plot(time, np.degrees(angVeldes[:,0]), lw=0.75, c='darkgreen',label='Reference')
    ax4[1].plot(time, np.degrees(angVeldes[:,1]), lw=0.75, c='darkgreen',label='Reference')
    ax4[2].plot(time, np.degrees(angVeldes[:,2]), lw=0.75, c='darkgreen',label='Reference')

    ax4[0].set_ylabel('wx [deg/s]',labelpad=-2), ax4[1].set_ylabel('wy [deg/s]',labelpad=-2), ax4[2].set_ylabel('wz [deg/s]',labelpad=-2)
    fig4.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(3,1)
    create_subtitle(fig3, grid[0, ::], 'Actual vs Reference Angular Velocities')

    ###################################

    ax5[0,0].plot(time, poserr[:,0],c='r',lw=0.7), ax5[0,1].plot(time, poserr[:,1],c='r',lw=0.7), ax5[0,2].plot(time, poserr[:,2],c='r',lw=0.7)
    ax5[0,0].set_ylabel('ex [m/s]'), ax5[0,1].set_ylabel('ey [m/s]'), ax5[0,2].set_ylabel('ez [m/s]')
    
    ax5[1,0].plot(time, linVerr[:,0],c='r',lw=0.7), ax5[1,1].plot(time, linVerr[:,1],c='r',lw=0.7), ax5[1,2].plot(time, linVerr[:,2],c='r',lw=0.7)
    ax5[1,0].set_ylabel('vex [m/s]'), ax5[1,1].set_ylabel('vey [m/s]'), ax5[1,2].set_ylabel('vez [m/s]')
    fig5.supxlabel(ts,fontsize='small')

    grid = plt.GridSpec(2,3)
    create_subtitle(fig5, grid[0, ::], 'Positional errors')
    create_subtitle(fig5, grid[1, ::], 'Linear Velocities errors')
    
    ###################################

    ax6.plot(time, cont_stack[:,0],lw=0.8, c='darkblue')
    ax6.set_ylabel('fz [N]')
    
    ax7.plot(time, cont_stack[:,1], lw=0.8, c='darkblue'), ax8.plot(time, cont_stack[:,2], lw=0.8, c='darkblue'), ax9.plot(time, cont_stack[:,3],lw=0.8, c='darkblue')
    ax7.set_ylabel('taux [N.m]',fontsize='small'), ax8.set_ylabel('tauy [N.m]',fontsize='small'), ax9.set_ylabel('tauz [N.m]',fontsize='small')
    fig6.supxlabel(ts,fontsize='small')

    create_subtitle(fig6, gs[::, 0], 'Force Control Input')
    create_subtitle(fig6, gs[::, 1], 'Torque Control Input')

    ###################################
    ax10.plot3D(pos[:,0], pos[:,1], pos[:,2], 'k-.',lw=1.5, label="Actual Trajectory")
    ax10.plot3D(posdes[:,0], posdes[:,1] , posdes[:,2],'darkgreen',ls='--',lw=1.5,label="Reference Trajectory")
    ax10.legend()
    ax10 = setlimits(ax10, pos)
   ################################### 
    if controller in 'lee':
        txt = 'Kp = [{:.2f},{:.2f},{:.2f}]\n \
        Kd = [{:.2f},{:.2f},{:.2f}]\n \
        KR = [{:.4f},{:.4f},{:.4f}]\n \
        Kw = [{:.4f},{:.4f},{:.4f}]'.format(Kpxy, Kpxy, Kpz, Kdxy, Kdxy, Kdz, Krxy, Krxy, Krz, Kwxy, Kwxy, Kwz)
        textfig, textax = plt.subplots(figsize=(3, 3))
        textax.grid(False)
        textax.axis(False)
        textax.text(0, 1, txt, size=13, color='black')
        textfig.savefig(f, format='pdf', bbox_inches='tight')
    fig1.savefig(f, format='pdf', bbox_inches='tight')
    fig2.savefig(f, format='pdf', bbox_inches='tight')
    fig3.savefig(f, format='pdf', bbox_inches='tight')        
    fig4.savefig(f, format='pdf', bbox_inches='tight')  
    fig5.savefig(f, format='pdf', bbox_inches='tight')  
    fig6.savefig(f, format='pdf', bbox_inches='tight')  
    fig7.savefig(f, format='pdf', bbox_inches='tight')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_usd")
    parser.add_argument("controller")
    parser.add_argument("filename")
    
    args = parser.parse_args()
    main(args)
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from geom import rotmat, delrotmat, H_between2, H_prior2, rotmatC, H_graph, H_graph_conv, solve_graph_convex, solve_graph
from convex import hull_so2, graph_hull_so2#, H_between2_conv

N = 301
T = np.arange(N)
pos = np.vstack((np.cos(T*2*np.pi/(N-1) - np.pi/2),np.sin(T*2*np.pi/(N-1) - np.pi/2),T*2*np.pi/(N-1)))

#meas_noise_xy = np.random.normal
#bot was always moving forward and rotating a little bit.
meas = {}
#add relative odom constraints (noiseless for now)
for i in range(N-1):
    th = (pos[2,i+1] - pos[2,i]    +np.pi)%(2*np.pi)-np.pi
    rel_trans = rotmat(pos[2,i]).T @ (pos[:2,i+1] - pos[:2,i])
    meas[('b',i,i+1)] = np.asarray([[rel_trans[0],rel_trans[1],th]]).T + np.random.normal(scale=0.01,size=((3,1)))
meas[('b',0,N-1)] = np.zeros((3,1))
meas[('p',0,)] = pos[:,0,np.newaxis]
rpos = np.vstack((pos[0,:],pos[1,:],np.cos(pos[2,:]),np.sin(pos[2,:])))

#xstart = 

posC = hull_so2(pos)
measC = graph_hull_so2(meas)

#direct gradient descent solution
pn = np.zeros((3,N))

#measurement loss
COV = np.diag([0.1,0.1,1.0])
CINV = np.linalg.inv(COV)



x0g = np.copy(pos) + np.random.normal(0.0,1.0,size=pos.shape)
x0 = hull_so2(x0g)

#x0g, lossesg, lamsg, statesg = solve_graph(x0g,meas,n_steps=10000)

x0, losses, lams, states = solve_graph_convex(x0,measC,n_steps=3000)

#x0g = np.vstack([x0[0,:],x0[1,:],np.arctan2(x0[3,:],x0[2,:])])

x0g, losses, lams, statesg = solve_graph(x0g, meas, n_steps = 3000)

final_angles = np.arctan2(x0[3,:],x0[2,:])

plt.subplot(1,2,1)
plt.plot(losses)
plt.yscale('log')
plt.subplot(1,2,2)
plt.plot(lams)
plt.yscale('log')
plt.figure()
plt.plot(pos[0,:],pos[1,:], 'k', label="Ground Truth Trajectory")
plt.plot(x0[0,:],x0[1,:],'orange', label="Estimated Trajectory (Relaxed))")
plt.plot(x0g[0,:],x0g[1,:],'b', label="Estimated Trajectory (LMQ)")
plt.grid()
#plt.legend()
plt.tight_layout()
plt.gca().set_aspect(1.0)
plt.figure()
plt.plot(pos[2,:],color='k')
plt.plot(final_angles, color='b')

'''
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect(1.0)
def animate(idx):
    ax.clear()
    ax.plot(pos[0,:],pos[1,:], 'k')
    #ax.plot(states[idx][0,:],states[idx][1,:], 'b')
    ax.plot(statesg[idx][0,:],statesg[idx][1,:], 'orange')
    ax.set_title(idx)
import matplotlib.animation as ani
anim = ani.FuncAnimation(fig, animate, frames=range(0,len(statesg),2))
plt.show()
'''

plt.show()


import numpy as np
import tqdm
import multiprocessing as mp
import time
def rotmat(th):
    c=np.cos(th)
    s=np.sin(th)
    return np.asarray([[c,s],[-s,c]])
def delrotmat(th):
    c=np.cos(th)
    s=np.sin(th)
    return np.asarray([[-s,c],[-c,-s]])

def rotmatC(cs):
    return np.asarray([[cs[0],cs[1]],[-cs[1],cs[0]]])

def delrotmatC(cs):
    return np.asarray([[-cs[1],cs[0]],[-cs[0],-cs[1]]])
    
def H_between2(xi,xj,meas,CINV=np.identity(3),no_jac = False):
    #loss

    delta = np.zeros((3,1))
    R = rotmat(xi[2]).T
    dR = delrotmat(xi[2]).T
    delta_pos = (xj[:2] - xi[:2])
    delta[:2,0] = (R @ delta_pos) - meas[:2,0]
    delta[2] = (xj[2] - xi[2] - meas[2] + np.pi) %(2*np.pi) - np.pi
    deltc = delta.T @ CINV
    if no_jac:
        return (deltc @ delta)[0,0]
    #loss jacobian
    del_i = np.zeros((1,3))
    del_j = np.zeros((1,3))

    del_H_i = np.zeros((3,3))
    del_H_i[:2,:2] = -R
    del_H_i[2,2] = -1
    del_H_i[:2,2] = dR @ delta_pos

    del_H_j = np.zeros((3,3))
    del_H_j[:2,:2] = R
    del_H_j[2,2] = 1

    return (deltc @ delta)[0,0], 2*deltc @ del_H_i, 2*deltc @ del_H_j

def H_prior2(xi,meas,CINV=np.identity(3),no_jac = False):
    #loss
    delta = np.zeros((3,1))
    delta[:2,0] = xi[:2] - meas[:2,0]
    delta[2] = (xi[2] - meas[2] + np.pi) %(2*np.pi) - np.pi

    deltc = delta.T @ CINV
    if no_jac:
        return (deltc @ delta)[0,0]
    #loss jacobian
    del_H_i = np.identity(3)
    return (deltc @ delta)[0,0], 2*deltc @ del_H_i

# full jacobian of measurement state
def H_graph(state, meas, CINV=np.identity(3), no_jac=False):
    loss = 0.0
    grad_H  = np.zeros_like(state)
    #tic = time.time()
    for m in meas:
        if m[0] == 'b':
            #pose-pose measurement
            if no_jac:
                loss += H_between2(state[:,m[1]],state[:,m[2]],meas[m],CINV=CINV,no_jac=no_jac)
            else:
                l, jac_i, jac_j = H_between2(state[:,m[1]],state[:,m[2]],meas[m],CINV=CINV,no_jac=no_jac)
                grad_H[:,m[1]] += jac_i[0]
                grad_H[:,m[2]] += jac_j[0]
                loss += l
        if m[0] == 'p':
            #prior
            if no_jac:
                loss += H_prior2(state[:,m[1]], meas[m], CINV=CINV,no_jac=no_jac)
            else:
                l, jac_i = H_prior2(state[:,m[1]], meas[m], CINV=CINV,no_jac=no_jac)
                grad_H[:,m[1]] += jac_i[0]
                loss += l
    #print(f"{time.time() - tic}")
    if no_jac:
        return loss
    else:
        return loss, grad_H



SO2_COS_INNER_DERIVATIVE = np.identity(2)
SO2_SIN_INNER_DERIVATIVE = np.asarray([[0,1],[-1,0]])

#before was roughly 100us per execution, trying to lower that.
def H_between2_conv(xi,xj,meas,alpha=1.0,beta=1.0, no_jac=False):
    #tic = time.time()
    RJ = rotmatC(xj[2:])
    RI = rotmatC(xi[2:])
    RIJ = rotmatC(meas[2:,0])
    DELTA_R = RJ - RIJ @ RI
    
    del_pos = xj[:2] - xi[:2] - RI @ meas[:2,0]

    loss = alpha*np.sum(DELTA_R*DELTA_R) + beta*del_pos.T @ del_pos

    if no_jac:
        return loss,0,0

    #import pdb
    #pdb.set_trace()
    #for faster version:
    '''
    gdel_Hc_j = np.zeros((4,))
    #derivative wrt xj, yj
    gdel_Hc_j[:2] = 2*beta*del_pos

    gdel_Rj_inner = 2*alpha*(RJ - RIJ @ RI)
    #derivative wrt cj
    gdel_Hc_j[2] = np.tensordot(gdel_Rj_inner, SO2_COS_INNER_DERIVATIVE)
    #derivative wrt sj
    gdel_Hc_j[3] = np.tensordot(gdel_Rj_inner, SO2_SIN_INNER_DERIVATIVE)
    '''
    del_pos *= 2*beta
    DELTA_R *= 2*alpha
    
    del_Hc_j = np.asarray([del_pos[0],del_pos[1],DELTA_R[0,0] + DELTA_R[1,1],DELTA_R[0,1]-DELTA_R[1,0]])
    #'''

    '''
    gdel_Hc_i = np.zeros((4,))
    #derivative wrt xi, yi
    gdel_Hc_i[:2] = -del_pos
    
    #derivative wrt ci
    #first term is from rotation loss, second is from translation
    trans_outer = np.outer(xi[:2]-xj[:2] + RI @ meas[:2,0],meas[:2,0])
    gdel_Ri_inner = 2*alpha*(RI - RIJ.T @ RJ) + 2*beta*trans_outer
                                  
    gdel_Hc_i[2] = np.tensordot(gdel_Ri_inner , SO2_COS_INNER_DERIVATIVE)
    #derivative wrt si
    gdel_Hc_i[3] = np.tensordot(gdel_Ri_inner, SO2_SIN_INNER_DERIVATIVE)
    '''
    #MDELTA_R = 2*alpha*(-RIJ.T @ DELTA_R)
    #trans_outer = 2*beta*np.outer(-del_pos,meas[:2,0])
    del_RI = (-RIJ.T @ DELTA_R) + np.outer(-del_pos,meas[:2,0])
    del_Hc_i = np.asarray([-del_pos[0],-del_pos[1],del_RI[0,0]+del_RI[1,1],del_RI[0,1]-del_RI[1,0]])
    #import pdb
    #pdb.set_trace()
    #print(f"{time.time() - tic}")

    return loss, del_Hc_i, del_Hc_j

def H_prior2_conv(xi,meas,alpha=1.0,beta=1.0,no_jac = False):
    RI = rotmatC(xi[2:])
    RP = rotmatC(meas[2:,0])
    del_pos = xi[:2] - meas[:2,0]
    del_R = RI - RP
    loss = beta*np.inner(del_pos,del_pos) + alpha*np.sum(del_R*del_R)
    del_pos *= 2*beta
    del_R *= 2*alpha
    if no_jac:
        return loss,0
    del_Hc_i = np.asarray([del_pos[0],del_pos[1],del_R[0,0]+del_R[1,1],del_R[0,1]-del_R[1,0]])
    #del_Hc_i = np.zeros((4,))
    #del_Hc_i[:2] = 2*del_pos
    #del_Hc_i[2] = 2*(del_R[0,0] + del_R[1,1])
    #del_Hc_i[3] = 2*(del_R[0,1] - del_R[1,0])

    return loss, del_Hc_i
    

#'''
def H_graph_conv(state, meas,alpha=1.0,beta=1.0, no_jac=False):
    loss = 0.0
    delta = np.zeros((state.shape))
    for m in meas:
        if m[0] == 'b':
            l, dHi, dHj =  H_between2_conv(state[:,m[1]],state[:,m[2]],meas[m],alpha,beta, no_jac=no_jac)
            loss += l
            delta[:,m[1]] += dHi
            delta[:,m[2]] += dHj
        if m[0] == 'p':
            l, dHi = H_prior2_conv(state[:,m[1]], meas[m], alpha, beta, no_jac=no_jac)
            loss += l
            delta[:,m[1]] += dHi
    return loss, delta
#'''
'''
CONV_POOL = mp.Pool()
def H_graph_conv(state, meas,alpha=1.0,beta=1.0, no_jac=False):
    #tic = time.time()
    loss = 0.0
    delta = np.zeros((state.shape))
    res = []

    for m in meas:
        if m[0] == 'b':
            res.append((m,CONV_POOL.apply_async(H_between2_conv, (state[:,m[1]],state[:,m[2]],meas[m],alpha,beta,no_jac))))
        if m[0] == 'p':
            res.append((m,CONV_POOL.apply_async(H_prior2_conv, (state[:,m[1]],meas[m],alpha,beta,no_jac))))
    toc = time.time()
    for m, re in res:
        if m[0] == 'b':
            l, dHi, dHj = re.get()
            loss += l
            delta[:,m[1]] += dHi
            delta[:,m[2]] += dHj
        if m[0] == 'p':
            l, dHi = re.get()
            loss += l
            delta[:,m[1]] += dHi
    #print(f"{toc - tic} {time.time() - toc}")
    return loss, delta

'''

def solve_graph_convex(x0, measC, alpha=1.0, beta=1.0, return_states=True,n_steps=2000):

    ybound = np.inf
    xsh = x0.shape
    lam = 1.0
    losses = []
    lams = []

    alpha=1.0
    beta=1.0

    if return_states:
        states = [x0]
    print(f"\nRelaxed: Optimizing {x0.shape[1]} poses with {len(measC.keys())} constraints\n")
    pbar = tqdm.trange(n_steps)
    for i in pbar:
        #tic = time.time()
        loss, grad = H_graph_conv(x0, measC)
        #print(f"{time.time() - tic}")
        x0 -= 1e-1*grad
        pbar.set_description(f"{loss:.3e}")
        losses.append(loss)
        continue
        #if i % 100 == 0.0:
        #    x0[2:,:] = x0[2:,:] / np.linalg.norm(x0[2:,:], axis=0)
        update_state = True

        while(update_state):

            if lam < 1e-20:
                break
            #I am taking the lambda term outside of the pinv as opposed to normal LMQ (so we decrease lambda on a failure) since I find it more natural to formulate the objective function this way.
            grad = grad.flatten()
            upd = x0 - (grad.T/(grad.T @ grad) * lam*loss).reshape(xsh)
            #upd = x0 - lam*grad

            ybound = H_graph_conv(upd, measC, no_jac=True)[0]

            #print(f"Tried lam {lam}, {ybound} vs {loss}")
            if ybound >= loss:
                lam *= 1e-1
            else:
                update_state = False
                lams.append(lam)
        x0 = upd
        
        #lam *= 1.01
        #print("succeeded")


        if return_states:
            states.append(np.copy(x0))
            
    print("")

    if return_states:
        return x0, losses, lams, states
    else:
        return x0, losses, lams

def solve_graph(x0,meas,CINV=np.identity(3),n_steps=1000,return_states=True):
    ybound = np.inf
    xsh = x0.shape
    lam = 1.0
    losses = []
    lams = []

    print(f"\nLMQ: Optimizing {x0.shape[1]} poses with {len(meas.keys())} constraints\n")

    
    if return_states:
        states = [x0]
    for i in tqdm.trange(n_steps):
        #tic = time.time()
        loss, grad = H_graph(x0, meas)
        #print(f"{time.time() - tic}")
        losses.append(loss)
        update_state = True
        while(update_state):

            if lam < 1e-20:
                break
            #I am taking the lambda term outside of the pinv as opposed to normal LMQ (so we decrease lambda on a failure) since I find it more natural to formulate the objective function this way. 
            upd = x0 - (np.linalg.pinv(grad.reshape((1,-1))) * lam*loss).reshape(xsh)
            ybound = H_graph(upd, meas, no_jac=True)

            #print(f"Tried lam {lam}, {ybound} vs {loss}")
            if ybound >= loss:
                lam *= 1e-1
            else:
                update_state = False
                lams.append(lam)
        lam *= 2
        #print("succeeded")

        x0 = upd
        if return_states:
            states.append(np.copy(x0))

    print("")

    if return_states:
        return x0, losses, lams, states
    else:
        return x0, losses, lams

#if __name__ == '__main__':
print("DO DERIVATIVE TEST")
k = np.linspace(-np.pi,np.pi,3000)
xi = np.random.normal(size=(4,))
#xj = np.random.normal(size=(4,))
meas = np.random.normal(size=(4,1))

f=np.zeros_like(k)
import matplotlib.pyplot as plt
for i in range(k.shape[0]):
    xi[3] = k[i]
    f[i] = H_prior2_conv(xi,meas,no_jac=True)[0]

pt = 1

xi[3] = pt
val, dI = H_prior2_conv(xi,meas)
slope = dI[3]
line = np.linspace(-1,1,1000)
plt.plot(line + pt, slope*line + val)

plt.plot(k,f)

plt.show()

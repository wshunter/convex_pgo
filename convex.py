import numpy as np

#project poses onto convex hull of se(2)
def hull_so2(pos):
    ret = np.zeros((pos.shape[0]+1,pos.shape[1]))
    ret[:2,:] = pos[:2,:]
    ret[2,:] = np.cos(pos[2,:])
    ret[3,:] = np.sin(pos[2,:])
    return ret

# project pose constraints onto convex hull of se(2)
def graph_hull_so2(graph):
    ret = {}

    for i in graph:
        new_meas = np.zeros((4,1))
        new_meas[:2,0] = graph[i][:2,0]
        new_meas[2,0] = np.cos(graph[i][2,0])
        new_meas[3,0] = np.sin(graph[i][2,0])
        ret[i] = np.copy(new_meas)
    return ret

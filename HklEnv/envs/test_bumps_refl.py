import bumps.names as bumps 
import bumps.fitters as fitters

import numpy as np

from .find_min import findmin

def better_bumps(model):

    zin=[]
    zout=[]
    chis=[]
    dzs=[]
    nllfs=[]

    for zs in np.arange(.05,.45,.005):
        #print("zs", zs)
        model.atomListModel.atomModels[0].z.value = zs
        model.update()
        schi=model.nllf()
        nllfs.append(schi)
        zin.append(zs)
    
    xpeaks = findmin(zin,nllfs,10)
    print('xpeak', xpeaks)
    model.atomListModel.atomModels[0].z.value = zin[xpeaks[0]]
    model.update()
    problem = bumps.FitProblem(model)

    result = fitters.fit(problem, method='lm')
    for p, v in zip(problem._parameters, result.dx):
        p.dx = v
    
    return result.x, result.dx, problem.chisq(), problem._parameters
    







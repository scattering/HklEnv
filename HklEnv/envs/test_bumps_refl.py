import os,sys;sys.path.append(os.path.abspath("/home/kmm11/pycrysfml/hklgen/"))
import bumps.names as bumps 
import bumps.fitters as fitters
import bumps.lsqerror as lsqerror
from bumps.formatnum import format_uncertainty_pm

import numpy as np

import fswig_hklgen as H
import hkl_model as Mod
import sxtal_model as S
import sys; sys.stdout.flush()
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axes

import random
from random import seed
from random import sample
from random import randint
from HklEnv.envs.find_min import findmin

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
    







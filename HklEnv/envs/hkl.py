import os
from os import path
from copy import copy
import random as rand
import pickle
import itertools

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axes

import gym
from gym.utils import seeding
from gym.spaces import Discrete
from baselines.spaces import Bin_Discrete

import bumps.names as bumps
import bumps.fitters as fitters
import bumps.lsqerror as lsqerror
from bumps.formatnum import format_uncertainty_pm

import hklgen
from hklgen import fswig_hklgen as H
from hklgen import hkl_model as Mod
from hklgen import sxtal_model as S

DATAPATH = os.environ.get('HKL_DATAPATH', None)
if DATAPATH is None:
    DATAPATH = os.path.join(os.path.abspath(os.path.dirname(hklgen.__file__)),
                            'examples', 'sxtal')

class HklEnv(gym.Env):

    def __init__(self, reward_scale=1e3, storspot="ppodat"):
        #self._first = True
        self.reward_scale=reward_scale
        print("Loading problem from %r. Set HklEnv.hkl.DATAPATH"
              " or os.environ['HKL_DATAPATH'] to override." % DATAPATH)
        observedFile = os.path.join(DATAPATH,r"prnio.int")
        infoFile = os.path.join(DATAPATH,r"prnio.cfl")

        print("look for storspot in available attrs", dir(self))
        print("args", getattr(self, 'args', None), getattr(self, 'extra_args', None))
        #Read data
        self.spaceGroup, self.crystalCell, self.atomList = H.readInfo(infoFile)

        #Return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
        wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=self.crystalCell)
        self.wavelength = wavelength
        self.refList = np.array(refList)
        self.sfs2 = sfs2
        self.error = error
        self.tt = [H.twoTheta(H.calcS(self.crystalCell, ref.hkl), wavelength) for ref in refList]
        self.backg = None
        self.exclusions = []
        self.hkls = []
        self.zs = []
        self.storspot = storspot
        self.envRank = 0
        self.repeats = 0
        self.repeatDeduction = 100000 #change me below!
        self.repeatDecay = 1
        self.epRepeats = []
        self.totReward = 0

        self.observation_space = Bin_Discrete(len(self.refList))
        self.action_space = Discrete(len(self.refList))

        self.episodeNum = 0
        
        self.reset()

    def epStep(self):
        self.episodeNum += 1
        

    def step(self, actions):
        #print("                                 stepping")
        #raise Hell
        reward = -self.reward_scale
        chisq = None
        dz =None
        repeatPunish = False
        self.steps += 1
        self.repeatDeduction = self.repeatDecay *  self.repeatDeduction
        for idx in range(0, len(self.visited)):
            if self.visited[idx] == self.refList[int(actions)]:
                print("repeat!")
                print("         curr reward:", self.totReward)
                
                
                self.repeats += 1
                #repeatPunish = True
                reward -= self.repeatDeduction
                print("         new reward:",self.totReward + reward)
                #return self.state, reward, False, {}
                break
        #No repeats
        self.visited.append(self.refList[int(actions)])
        self.state[int(actions)] = 1
        #print(actions)
        #self.remainingActions.remove(actions)
       

        #Find the data for this hkl value and add it to the model
        self.model.refList = H.ReflectionList(self.visited)
        self.model._set_reflections()
        
        self.model.error.append(self.error[actions])
        self.model.tt = np.append(self.model.tt, [self.tt[actions]])

        self.observed.append(self.sfs2[actions])
        self.model._set_observations(self.observed)
        self.model.update()
        self.hkls.append(self.refList[actions.tolist()].hkl)
        self.zs.append(self.model.atomListModel.atomModels[0].z.value)
        #reward = -self.reward_scale
        #print('reward',reward)
        #print(str(self.model))
        #Need more data than parameters, have to wait to the second step to fit
        if len(self.visited) > 2:
            #print(" about to fit")
            
            try:
                x, dx, chisq, params = self.fit(self.model)
            except ValueError:
                print('FAILLLLLLL    ',self.model.refList)
                print('actions', actions)
                print('visted:                  ', self.hkls)
                print('visted length (actions???):                  ', len(self.hkls))
                pass
            #'name': 'Pr z'
            dz=params[0].dx
            if self.prevDx != None and dz < self.prevDx:
                reward+=1/dz
                #print('reward',dz)
                
            self.prevDx=dz

            #if (self.prevChisq != None and chisq < self.prevChisq):
            #    reward = 1/(chisq*10)

            #self.prevChisq = chisq
        
        
                
        self.totReward += reward
        
        
        def snapshot():
            path = self.storspot if self.storspot else "."
            filename = "hklLog-%d_%d.txt" % (self.episodeNum, self.envRank) + ".txt"
            print("saving to", filename)
            with open(os.path.join(path, filename), "w+") as fid:
                file.write(str(self.hkls))
            filename = "zLog-%d_%d.txt" % (self.episodeNum, self.envRank) + ".txt"
            np.savetxt(os.path.join(path, filename), self.zs)
            filename = "repeats-%d.txt" % self.envRank + ".txt"
            np.savetxt(os.path.join(path, filename), self.epRepeats)

        if self.prevChisq is not None and len(self.visited) > 50 and chisq < 5:
            self.episodeNum += 1
            self.epRepeats.append(self.repeats)
            snapshot()
            return self.state, 1, True, {"chi": self.prevChisq, "z": self.model.atomListModel.atomModels[0].z.value, "hkl": self.refList[actions].hkl}

        if len(self.remainingActions) == 0 or self.steps > 100:
            terminal = True
            self.episodeNum += 1
            self.epRepeats.append(self.repeats)
            snapshot()
        elif repeatPunish:
            terminal = True
            self.episodeNum += 1
            self.epRepeats.append(self.repeats)
            snapshot()
        else:
            terminal = False

        return self.state, reward, terminal, {"chi": chisq, "z": self.model.atomListModel.atomModels[0].z.value, "hkl": self.refList[actions.tolist()].hkl} #, chisq, self.model.atomListModel.atomModels[0].z.value, self.refList[actions]

    def reset(self):
        #print("resetting")
        #Make a cell
        cell = Mod.makeCell(self.crystalCell, self.spaceGroup.xtalSystem)

        #Define a model
        self.model = S.Model([], [], self.backg, self.wavelength, self.spaceGroup, cell,
                             self.atomList, self.exclusions,
                    scale=0.06298, error=[],  extinction=[0.0001054])

        #Set a range on the x value of the first atom in the model
        self.model.atomListModel.atomModels[0].z.value = 0.25
        self.model.atomListModel.atomModels[0].z.range(0,0.5)

        self.visited = []
        self.observed = []
        self.hkls = []
        self.zs = []
        self.repeatDeduction = 100000
        self.repeats = 0
        self.remainingActions = []
        self.totReward = 0
        
        for i in range(len(self.refList)):
            self.remainingActions.append(i)

        self.totReward = 0
        self.prevChisq = None
        self.prevDx= None
        self.steps = 0

        self.state = np.zeros(len(self.refList))
        self.stateList = []
        #print("reset")
        return self.state

    def giveRank(self, subrank):
        self.envRank = subrank
    
    def fit(self, model):

        #Create a problem from the model with bumps,
        #then fit and solve it
        #print("fitting?  I HOPEEEEEEE")
        problem = bumps.FitProblem(model)
        result = fitters.fit(problem, method='lm')
        for p, v in zip(problem._parameters, result.dx):
            p.dx = v
        return result.x, result.dx, problem.chisq(), problem._parameters
        
        """   # Dead code
        fitted = fitters.LevenbergMarquardtFit(problem)
        x, fx = fitted.solve()
        cov = fitted.cov()
        dx = lsqerror.stderr(cov)
        #problem.setp(x)  <=== this is done already in LM (and all other fitters)
        #print('x,dx',x,dx, problem._parameters[0].__dict__)
        for p, v in zip(problem._parameters, dx):
            p.dx = v
        return x, dx, problem.chisq(),problem._parameters
        """

    @property
    def states(self):
        return dict(shape=self.state.shape, type='float')

    @property
    def actions(self):
        return dict(num_actions=len(self.refList), type='int')


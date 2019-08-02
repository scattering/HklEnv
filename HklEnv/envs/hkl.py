import os,sys;sys.path.append(os.path.abspath("/home/kmm11/pycrysfml/hklgen/"))

from os import path
import os
import gym 
from baselines import spaces as base_spaces 
from gym import spaces as gym_spaces 
from gym.utils import seeding
from copy import copy
import numpy as np
import random as rand
import pickle
import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axes

import bumps.names as bumps
import bumps.fitters as fitters
import bumps.lsqerror as lsqerror
from bumps.formatnum import format_uncertainty_pm

import fswig_hklgen as H
import hkl_model as Mod
import sxtal_model as S
#from tensorforce.environments import Environment

from HklEnv.envs.test_bumps_refl import better_bumps

def profile(fn, *args, **kw):
    """
    Profile a function called with the given arguments.
    """
    import cProfile
    import pstats

    print("in profile", fn, args, kw)
    result = [None]
    def call():
        try:
            result[0] = fn(*args, **kw)
        except BaseException as exc:
            result.append(exc)
    datafile = 'profile.out'
    cProfile.runctx('call()', dict(call=call), {}, datafile)
    stats = pstats.Stats(datafile)
    # order='calls'
    order = 'cumulative'
    # order='pcalls'
    # order='time'
    stats.sort_stats(order)
    stats.print_stats()
    os.unlink(datafile)
    if len(result) > 1:
        raise result[1]
    return result[0]
    
class Profiler(object):
    def __init__(self,  fn, datafile='profile.out'):
        self.fn = fn
        self.datafile = datafile
        self.first = True
        
    def __call__(self, *args, **kw):
        #print("in call", self, args, kw, self.fn)
        if self.first:
            self.first = False
            import cProfile
            
            result = [None]
            def call():
                result[0] = self.fn(fn, *args, **kw)
            
            cProfile.runctx('call()', dict(call=call), {}, self.datafile)
            self.summarize()
            return result[0]
        else:
            return self.fn(*args, **kw)
            
    def summarize(self):
        """
        Profile a function called with the given arguments.
        """
        import pstats, sys

        with open("stats.out", "w") as stream:
            stats = pstats.Stats(self.datafile, stream=stream)
            # order='calls'
            order = 'cumulative'
            # order='pcalls'
            # order='time'
            stats.sort_stats(order)
            stats.print_stats()
        
    def cleanup():
        self.summarize()
        os.unlink(self.datafile)


class HklEnv(gym.Env):

    def __init__(self, reward_scale=1e2, storspot = "ppodat"):
        #self._first = True
        self.reward_scale=reward_scale
        print("envmade")
        import sys; sys.stdout.flush()
        DATAPATH = os.path.abspath(os.path.expanduser("~/pycrysfml/hklgen/examples/sxtal"))
        observedFile = os.path.join(DATAPATH,r"prnio.int")
        infoFile = os.path.join(DATAPATH,r"prnio.cfl")

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
        self.hs = []
        self.ks = []
        self.ls = []
        self.storspot = "/wrk/kmm11/" + storspot
        self.envRank = 0
        self.repeats = 0
        self.repeatDeductionTemp = 100000
        self.repeatDeduction = self.repeatDeductionTemp #change me below!
        self.repeatDecay = 1
        self.epRepeats = []
        self.totReward = 0
        self.rewards = []
        self.observation_space = base_spaces.Bin_Discrete(len(self.refList))
        self.action_space = gym_spaces.Discrete(len(self.refList))
        
        self.batch = 0
        self.stepNum = 0
        self.valid_actions = np.ones(shape = (5, len(self.refList)))
        
        self.remaining_acs = np.zeros(198)
        for i in range (0, 198):
            self.remaining_acs[i] = i

        self.episodeNum = 0
        
        self.reset()
        

    def step(self, actions):
        print("stepping", actions)
        small_scale_ac = actions
        actions = self.remaining_acs[int(actions)]  
        self.remaining_acs = np.delete(self.remaining_acs, small_scale_ac)
        self.stepNum += 1
        self.batch = self.stepNum % 5
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
                #print("         curr reward:", self.totReward)
                
                
                self.repeats += 1
                #repeatPunish = True
                reward -= self.repeatDeduction
                #print("         new reward:",self.totReward + reward)
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
        
        self.model.error.append(self.error[int(actions)])
        self.model.tt = np.append(self.model.tt, [self.tt[int(actions)]])

        self.observed.append(self.sfs2[int(actions)])
        self.model._set_observations(self.observed)
        self.model.update()
        self.hkls.append(self.refList[int(actions)].hkl)
        self.zs.append(self.model.atomListModel.atomModels[0].z.value)
        #print('reward',reward)
        #print(str(self.model))
        #Need more data than parameters, have to wait to the second step to fit
        #print("state                                                   ", self.state)
        if len(self.visited) > 1:
            #print(" about to fit")
            
            #print('visted:                  ', self.hkls)
            #print("repeats:                 ", self.repeats)
            
            if len(self.visited) == (self.repeats + 1) :
                return self.state, reward, False, {}
            else :
                x, dx, chisq, params = better_bumps(self.model)
            
            """
            try:
                x, dx, chisq, params = self.fit(self.model)
            except ValueError:
                print('FAILLLLLLL    ',self.model.refList)
                print('actions', actions)
                print('visted:                  ', self.hkls)
                print('visted length (actions???):                  ', len(self.hkls))
                pass
            """
            
            #'name': 'Pr z'
            dz=params[0].dx
            #reward += 1/abs(x - 0.35973)[0]
            if chisq <10:
                reward += 1000
                if dz < 1e-4:
                    reward+=1/dz
            self.prevDx=dz
            
            #if (self.prevChisq != None and chisq < self.prevChisq):
            #    reward = 1/(chisq*10)

            #self.prevChisq = chisq
        
        
                
        self.totReward += reward
        self.rewards.append(self.totReward)
        
        hkls_arr = np.asarray(self.refList[int(actions)].hkl)
        self.hs.append(hkls_arr[0])
        self.ks.append(hkls_arr[1])
        self.ls.append(hkls_arr[2])
        
        #print("batch: ", self.batch)
        #print("valid actions[batch]: ", self.valid_actions[self.batch])
        for i in range(len(self.valid_actions)):
            self.valid_actions[i] = (self.state +1) % 2
        
        #print("valid actions[batch] after: ", self.valid_actions[self.batch])
        #print("valid actions: ", self.valid_actions)
        
        print("invalid actions in environment are: ")
        counter = 0
        invalid_actions = []
        for i in range (0, len(self.valid_actions[self.batch])):
            if self.valid_actions[self.batch][i] == 0:
                invalid_actions.append(i)
            counter += 1
        print(invalid_actions)
        print("")
        
        print("valid actions:           ", invalid_actions)
        
        if (self.prevChisq != None and len(self.visited) > 50 and chisq < 5):
            self.episodeNum += 1
            self.log()
            return self.state, 1, True, {"chi": self.prevChisq, "z": self.model.atomListModel.atomModels[0].z.value, "hkl": self.refList[int(actions)].hkl, 'valid_actions': self.valid_actions}
        if (len(self.remainingActions) == 0 or self.steps > 30):
            terminal = True
            self.episodeNum += 1
            self.log()
        else:
            terminal = False

        return self.state, reward, terminal, {"chi": chisq, "z": self.model.atomListModel.atomModels[0].z.value, "hkl": self.refList[int(actions)].hkl, 'valid_actions': self.valid_actions} #, chisq, self.model.atomListModel.atomModels[0].z.value, self.refList[actions]

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
        self.hs = []
        self.ks = []
        self.ls = []
        self.valid_actions = np.ones(shape = (5, len(self.refList)))
        self.repeatDeductionTemp
        self.repeats = 0
        self.remainingActions = []
        self.totReward = 0
        self.stepNum = 0 
        
        self.remaining_acs = np.zeros(198)
        for i in range (0, 198):
            self.remaining_acs[i] = i
        
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
        
    def giveBatch(self, batch):
        self.batch = batch
        
    def log(self):
        #filename = self.storspot +"/hklLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        #np.savetxt(filename, self.hkls)
        file = open(self.storspot +"/hklLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt", "w+")
        file.write(str(self.hkls))
        file.close()
        
        filename = self.storspot + "/zLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        np.savetxt(filename, self.zs)
        
        filename = self.storspot + "/repeats-" + str(self.envRank) + ".txt"
        self.epRepeats.append(self.repeats)
        np.savetxt(filename, self.epRepeats)
        
        filename = self.storspot +"/hLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        np.savetxt(filename, self.hs)
        
        filename = self.storspot +"/kLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        np.savetxt(filename, self.ks)
        
        filename = self.storspot +"/lLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        np.savetxt(filename, self.ls)
        
        print("ENDED EPISODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        if self.totReward < 1e8:
            self.rewards.append(self.totReward)
            filename = self.storspot +"/rewardLog-" +  str(self.envRank) + ".txt"
            np.savetxt(filename, self.rewards)      
        
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




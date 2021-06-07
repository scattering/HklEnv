import sys

import numpy as np
import matplotlib as mpl
import bumps.names as bumps
from bumps import fitters

mpl.use('Agg')

sys.path.append("/home/kaet/gen/nist/pycrysfml")

from hklgen import fswig_hklgen as H
from hklgen import hkl_model as Mod
from hklgen import sxtal_model as S

from test_bumps_refl import better_bumps, graph

def step(actions):
    visited.append(refList[int(actions)])

    model.refList = H.ReflectionList(visited)
    model._set_reflections()

    model.error.append(error[int(actions)])
    model.tt = np.append(model.tt, [tt[int(actions)]])

    observed.append(sfs2[int(actions)])
    model._set_observations(observed)
    model.update()
    print(actions)
    if (len(visited) > 3):
        x, dx, chisq, params = better_bumps(model)
        print("dx:", dx)

# TODO: unhardcode if sharing code
observedFile = "/home/kaet/gen/nist/pycrysfml/hklgen/examples/sxtal/prnio.int"
infoFile = "/home/kaet/gen/nist/pycrysfml/hklgen/examples/sxtal/prnio.cfl"

# Read data
spaceGroup, crystalCell, atomList = H.readInfo(infoFile)

# Return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=crystalCell)
wavelength = wavelength
refList = np.array(refList)
sfs2 = sfs2
error = error
tt = [H.twoTheta(H.calcS(crystalCell, ref.hkl), wavelength) for ref in refList]
backg = None
exclusions = []
visited = []
observed = []

# from reset()
cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)

# Define a model
model = S.Model([], [], backg, wavelength, spaceGroup, cell,
                     atomList, exclusions,
                     scale=0.06298, error=[], extinction=[0.0001054])

# Set a range on the x value of the first atom in the model
# model.atomListModel.atomModels[0].z.value = 0.25
# model.atomListModel.atomModels[0].z.range(0, 0.45)

# model.atomListModel.atomModels[5].x.value = 0
# model.atomListModel.atomModels[5].x.range(0, 0.4)
#
# # model.atomListModel.atomModels[5].y = model.atomListModel.atomModels[5].x
# model.atomListModel.atomModels[5].y.value = 0
# model.atomListModel.atomModels[5].y.range(0, 0.4)
#
# model.atomListModel.atomModels[5].z.value = 0.17349
# model.atomListModel.atomModels[5].z.range(0.16, 0.18)

model.atomListModel.atomModels[3].z.value = 0.2
model.atomListModel.atomModels[3].z.range(0, 0.5)

model.atomListModel.atomModels[3].occ.value = 0
model.atomListModel.atomModels[3].occ.range(1.0, 2.0)

model.atomListModel.atomModels[4].occ.value = 0
model.atomListModel.atomModels[4].occ.range(0, 1.0)

model.atomListModel.atomModels[5].x.value = 0.2
model.atomListModel.atomModels[5].x.range(0, 0.5)
model.atomListModel.atomModels[5].y.value = 0.2
model.atomListModel.atomModels[5].y.range(0, 0.5)
model.atomListModel.atomModels[5].z.value = 0.2
model.atomListModel.atomModels[5].z.range(0, 0.5)

model.atomListModel.atomModels[5].occ.value = 0
model.atomListModel.atomModels[5].occ.range(0, 1.0)

loadRefl = False

model.update()

actions = np.zeros(198)

if loadRefl == True:
    for i in range(198):
        # step(np.random.randint(0, 198)) # randomly pick action
        step(i)
else:
    actions = np.arange(0, 198, 1)

    for action in actions:
        visited.append(refList[int(action)])

    model.refList = H.ReflectionList(visited)
    model._set_reflections()

    for action in actions:
        model.error.append(error[int(action)])
        model.tt = np.append(model.tt, [tt[int(action)]])

    for action in actions:
        observed.append(sfs2[int(action)])

    model._set_observations(observed)
    model.update()

    problem = bumps.FitProblem(model)
    result = fitters.fit(problem, method='dream', store="/home/kaet/gen/nist/HklEnv/HklEnv/envs")
    for p, v in zip(problem._parameters, result.dx):
        p.dx = v

    print("1)")
    print("result:", result.x)
    print("uncertainty:", result.dx)
    print("chisq:", problem.chisq())
    print("params:", problem._parameters)
    print()

graph(model)


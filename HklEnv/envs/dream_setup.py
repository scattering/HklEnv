import sys

import numpy as np
import bumps.names as bumps

sys.path.append("/home/kaet/gen/nist/pycrysfml")

from hklgen import fswig_hklgen as H
from hklgen import hkl_model as Mod
from hklgen import sxtal_model as S

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
model.atomListModel.atomModels[3].z.value = 0.2
model.atomListModel.atomModels[3].z.range(0, 0.5)

model.atomListModel.atomModels[3].occ.value = 0
model.atomListModel.atomModels[3].occ.range(1.8, 3.0)

model.atomListModel.atomModels[4].occ.value = 0.3
model.atomListModel.atomModels[4].occ.range(0, 2.0)

model.atomListModel.atomModels[5].x.value = 0.2
model.atomListModel.atomModels[5].x.range(0, 0.5)
model.atomListModel.atomModels[5].y.value = 0.2
model.atomListModel.atomModels[5].y.range(0, 0.5)
model.atomListModel.atomModels[5].z.value = 0.1
model.atomListModel.atomModels[5].z.range(-0.2, 0.3)

model.atomListModel.atomModels[5].occ.value = 0
model.atomListModel.atomModels[5].occ.range(.8, 8.0)

model.update()

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

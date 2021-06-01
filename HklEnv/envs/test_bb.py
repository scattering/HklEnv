import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("/home/kaet/gen/nist/pycrysfml")
print(sys.path)

from hklgen import fswig_hklgen as H
from hklgen import hkl_model as Mod
from hklgen import sxtal_model as S

from test_bumps_refl import better_bumps

def step(action):
    visited.append(refList[int(actions)])

    model.refList = H.ReflectionList(visited)
    model._set_reflections()

    model.error.append(error[int(actions)])
    model.tt = np.append(model.tt, [tt[int(actions)]])

    observed.append(sfs2[int(actions)])
    model._set_observations(observed)
    model.update()

    print(better_bumps(model))

if __name__ == "main":
    # TODO: unhardcode if sharing code
    observedFile = "/home/kaet/gen/nist/pycrysfml/hklgen/examples/sxtal/prnio.int"
    infoFile = "/home/kaet/gen/nist/pycrysfml/hklgen/examples/sxtal/prnio.cfl"

    # Read data
    spaceGroup, crystalCell, atomList = H.readInfo(infoFile)

    # Return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
    wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=crystalCell)
    print(error)
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
    model.atomListModel.atomModels[0].z.value = 0.25
    model.atomListModel.atomModels[0].z.range(0, 0.45)

    model.atomListModel.atomModels[5].x.value = 0.1
    model.atomListModel.atomModels[5].x.range(0, 0.2)

    model.atomListModel.atomModels[5].y.value = 0.1
    model.atomListModel.atomModels[5].y.range(0, 0.2)

    model.update()

    actions = np.zeros(198)

    for i in range(8):
        step(np.random.randint(0, 198)) # randomly pick action

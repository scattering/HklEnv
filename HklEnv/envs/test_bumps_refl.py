import matplotlib.pyplot as plt
import numpy as np

import bumps.names as bumps
import bumps.fitters as fitters


def better_bumps(model):
    model.atomListModel.atomModels[0].z.value = .35973
    model.atomListModel.atomModels[5].x.value = .07347
    model.atomListModel.atomModels[5].y.value = .07347
    model.update()
    problem = bumps.FitProblem(model)

    result = fitters.fit(problem, method='lm')
    for p, v in zip(problem._parameters, result.dx):
        p.dx = v

    return result.x, result.dx, problem.chisq(), problem._parameters


def chi_surface(model):
    xin = []
    yin = []
    chis = []

    for xs in np.arange(.05, .45, .01):
        for ys in np.arange(.05, .45, .01):
            model.atomListModel.atomModels[5].x.value = xs
            model.atomListModel.atomModels[5].y.value = ys
            model.update()
            xin.append(xs)
            yin.append(ys)
            try:
                problem = bumps.FitProblem(model)
            except:
                print("ERROR AT:" + xs)

            result = fitters.fit(problem, method='lm')
            for p, v in zip(problem._parameters, result.dx):
                p.dx = v

            chis.append(problem.chisq())

    return xin, yin, chis

def graph(model):
    xin, yin, chis = chi_surface(model)
    ax = plt.axes(projection='3d')
    ax.plot(xin, yin, chis, 'bo')
    plt.axvline(x=0.07347)
    plt.ylabel("chi squared")
    plt.xlabel("Oy estimated value")
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/test.png")
    plt.close()

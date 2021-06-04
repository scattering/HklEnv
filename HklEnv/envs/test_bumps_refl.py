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
    xs = []
    dxs = []
    xs_nans = []
    chis_nans = []

    for x in np.arange(0, 0.5, .05):
        for y in np.arange(0, .5, .01):
            model.atomListModel.atomModels[5].x.value = x
            model.atomListModel.atomModels[5].y.value = y
            model.update()
            xin.append(x)
            yin.append(y)
            try:
                problem = bumps.FitProblem(model)
            except:
                print("ERROR AT:" + x)

            chi = problem.chisq()
            result = fitters.fit(problem, method='lm')
            for p, v in zip(problem._parameters, result.dx):
                p.dx = v
            nans = np.isnan(result.dx)

            if True in nans:
                xs_nans.append(result.x[0])
                chis_nans.append(chi)

            else:
                dxs.append(result.dx[0])
                xs.append(x)
                chis.append(chi)


    return xin, yin, chis, nans, chis_nans, xs, dxs, xs_nans

def graph(model):
    xin, yin, chis, nans, chis_nans, xs, dxs, xs_nans = chi_surface(model)
    #ax = plt.axes(projection='3d')

    plt.plot(yin, chis, 'bo')
    plt.axvline(x=0.07347)
    plt.ylabel("chi squared")
    plt.xlabel("Ox estimated value")
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/test.png")
    plt.close()

    # print("xs:", xs)
    # print("chis:", chis)
    plt.scatter(xs, chis, c='blue')
    plt.scatter(xs_nans, chis_nans, c='red')
    plt.xlabel("oxygen x value")
    plt.ylabel("chi")
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/testx.png")
    plt.close()

    plt.scatter(xs, dxs, c='blue')
    for x_nan in xs_nans:
        plt.axvline(x=x_nan, c='red')
    plt.xlabel("oxygen x value")
    plt.ylabel("uncertainty")
    plt.title("Oxygen x uncertainty")
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/x_uncertainty.png")
    plt.close()

    no_x_idx = []
    for i in range(len(xs)):
        if dxs[i] > 1e7:
            no_x_idx.append(i)

    new_xs = np.delete(xs, no_x_idx)
    new_dxs = np.delete(dxs, no_x_idx)
    plt.scatter(new_xs, new_dxs, c='blue')
    for x_nan in xs_nans:
        plt.axvline(x=x_nan, c='red')
    plt.xlabel("oxygen x value")
    plt.ylabel("uncertainty")
    plt.title("Oxygen x uncertainty without 1e8 values")
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/x_new_uncertainty.png")
    plt.close()

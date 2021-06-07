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
    ys = []
    ys_nans = []
    chis = []
    xs = []
    dxs = []
    xs_nans = []
    chis_nans = []

    for x in np.arange(0, 0.5, .02):
        for y in np.arange(0, .5, .02):
            model.atomListModel.atomModels[5].x.value = x
            model.atomListModel.atomModels[5].y.value = y
            model.update()
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
                ys_nans.append(result.x[1])
                chis_nans.append(chi)

            else:
                dxs.append(result.dx[0])
                xs.append(x)
                ys.append(y)
                chis.append(chi)


    return chis, nans, chis_nans, xs, dxs, xs_nans, ys, ys_nans

def graph(model):
    chis, nans, chis_nans, xs, dxs, xs_nans, ys, ys_nans = chi_surface(model)

    ax = plt.axes(projection='3d')
    ax.plot(ys, xs, chis, 'bo')
    plt.scatter(ys_nans, xs_nans, chis_nans, c='red')
    print(xs_nans)
    print(ys_nans)
    plt.ylabel("chi squared")
    plt.xlabel("Ox estimated value")
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/chisurface.png")
    plt.close()

    plt.plot(ys, chis, 'bo')
    plt.scatter(ys_nans, chis_nans, c='red')
    plt.axvline(x=0.07347)
    plt.ylabel("chi squared")
    plt.xlabel("Oy estimated value")
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/ychi.png")
    plt.close()

    # print("xs:", xs)
    # print("chis:", chis)
    plt.scatter(xs, chis, c='blue')
    plt.scatter(xs_nans, chis_nans, c='red')
    plt.axvline(x=0.07347)
    plt.xlabel("oxygen x value")
    plt.ylabel("chi")
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/xchi.png")
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
    plt.savefig("/home/kaet/gen/nist/HklEnv/HklEnv/envs/x_uncertainty_no1e8.png")
    plt.close()

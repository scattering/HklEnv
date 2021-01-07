import bumps.names as bumps
import bumps.fitters as fitters

def better_bumps(model):
    model.update()
    problem = bumps.FitProblem(model)

    result = fitters.fit(problem, method='dream')
    for p, v in zip(problem._parameters, result.dx):
        p.dx = v

    return result.x, result.dx, problem.chisq(), problem._parameters








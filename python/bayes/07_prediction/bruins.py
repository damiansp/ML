import math
import sys

import numpy as np
import scipy

sys.path.append('..')
from thinkbayes import MakeGaussianPmf, MakeMixture, Pmf, Suite


MAX_LIKELY_GOALS = 10


def MakeGaussianPmf(mu, sigma, n_sds, n=101):
    pmf = Pmf()
    low = mu - n_sds*sigma
    high = mu + n_sds*sigma
    for x in np.linspace(low, high, n):
        p = scipy.stats.norm.pdf(mu, sigma, x)
        pmf.Set(x, p)
    pmf.Normalize()
    return pmf


class Hockey(Suite):
    def __init__(self, name):
        self.name = name
        pmf = MakeGaussianPmf(2.7, 0.3, 4)
        Suite.__init__(self, pmf)

    def Likelihood(self, data, hypo):
        lam = hypo
        k = data
        like = EvalPoissonPmf(lam, k)
        return like


# In thinkbayes
def EvalPoissonPmf(lam, k):
    return (lam)**k * math.exp(-lam) / math.factorial(k)


# In thinkbayes
def EvalExponentialPdf(lam, x):
    return lam * math.exp(-lam * x)

# In thinkbayes
def MakePoissonPmf(lam, high):
    pmf = Pmf()
    for k in range(0, high + 1):
        p = EvalPoissonPmf(lam, k)
        pmf.Set(k, p)
    pmf.Normalize()
    return pmf

def MakeGoalPmf(suite):
    metapmf = Pmf()
    for lam, prob in suite.Items():
        pmf = MakePoissonPmf(lam, MAX_LIKELY_GOALS)
        metapmf.Set(pmf, prob)
    mix = MakeMixture(metapmf)
    return mix


suite1 = Hockey('bruins')
suite1.UpdateSet([0, 2, 8, 4])
suite2 = Hockey('canucks')
suite2.UpdateSet([1, 3, 1, 0])
lam = 3.4
goal_dist = MakePoissonPmf(lam, MAX_LIKELY_GOALS)

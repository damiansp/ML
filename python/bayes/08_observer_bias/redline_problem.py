import sys

sys.path.append('..')
from thinkbayes import MakeMixture, Pmf


def BiasPmf(pmf):
    new_pmf = pmf.Copy()
    for x, p in pmf.Items():
        new_pmf.Mult(x, x)
    new_pmf.Normalize()
    return new_pmf


def PmfOfWaitTime(pmf_zb):
    metapmf = Pmf()
    for gap, prob in pmf_zb.Items():
        uniform = MakeUniformPmf(0, gap)
        metapmf.Set(uniform, prob)
    pmf_y = MakeMixture(metapmf)
    return pmf_y


def MakeUniformPmf(low, high):
    pmf = Pmf()
    for x in MakeRange(low=low, high=high):
        pmf.Set(x, 1)
    pmf.Normalize()
    return pmf


def MakeRange(low, high, skip=10):
    return range(low, high + skip, skip)


class WaitTimeCalculator:
    def __init__(self, pmf_z):
        self.pmf_z = pmf_z
        self.pmf_zb = BiasPmf(pmf)
        self.pmf_y = self.PmfOfWaitTime(self.pmf_zb)
        self.pmf_x = self.pmf_y


wtc = WaitTimeCalculator(pmf_z)

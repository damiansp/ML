import sys

sys.path.append('..')
from thinkbayes import EstimatedPdf, MakeCdfFromlist, MakeMixture, MakePmfFromList, Pmf


UPPER_BOUND = 1200 # max hypothetical time between trains
OBSERVED_GAP_TIMES = [
    428.0, 705.0, 407.0, 465.0, 433.0, 425.0, 204.0, 506.0, 143.0, 351.0, 
    450.0, 598.0, 464.0, 749.0, 341.0, 586.0, 754.0, 256.0, 378.0, 435.0, 
    176.0, 405.0, 360.0, 519.0, 648.0, 374.0, 483.0, 537.0, 578.0, 534.0, 
    577.0, 619.0, 538.0, 331.0, 186.0, 629.0, 193.0, 360.0, 660.0, 484.0, 
    512.0, 315.0, 457.0, 404.0, 740.0, 388.0, 357.0, 485.0, 567.0, 160.0, 
    428.0, 387.0, 901.0, 187.0, 622.0, 616.0, 585.0, 474.0, 442.0, 499.0, 
    437.0, 620.0, 351.0, 286.0, 373.0, 232.0, 393.0, 745.0, 636.0, 758.0]


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


n = 220
cdf_z = MakeCdfFromList(OBSERVED_GAP_TIMES)
sample_z = cdf_z.Sample(n)
pmf_z = MakePmfFromList(sample_z)

# compute biased pmf and add some long delays
cdf_zp = BiasPmf(pmf_z).MakeCdf()
sample_zb = cdf.zp.Sample(n) + [1800, 2400, 3000]

# smooth distrib of zb and make wtc
pdf_zb = EstimatePdf(sample_zb)
xs = MakeRange(low=60)
pmf_zb = pdf_zb.MakePmf(xs)

# unbias distrib of zb, make wtc
pmf_z = UnbiasPmf(pmf_zb)
wtc = WaitTimeCalculator(pmf_z)

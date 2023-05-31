'''
A railroad numbers its locomotives 1...N.  One day you see a locomotive with the
number 60. Estimate how many locomotives the railroad has.
'''
import sys

import matplotlib.pyplot as plt

sys.path.append('..')
from thinkbayes import Pmf, Suite


# prior
hypos = range(1, 1001)

class Train1(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1. / hypo

suite = Train1(hypos)
observed = 60
suite.Update(observed)
assignments = sorted(suite.Items())
hs, ps = [a[0] for a in assignments], [a[1] for a in assignments]

plt.plot(hs, ps, 'k-')
plt.xlabel('No. Trains')
plt.ylabel('Prob.')
plt.show()

def Mean(suite):
    total = 0
    for h, p in suite.Items():
        total += h * p
    return total

print('Mean:', Mean(suite))
print(suite.Mean()) # built-in method


observed = [30, 90]
for obs in observed:
    suite.Update(obs)
assignments2 = sorted(suite.Items())
hs2, ps2 = [a[0] for a in assignments2], [a[1] for a in assignments2]

plt.plot(hs2, ps2, 'k-')
plt.xlabel('No. Trains')
plt.ylabel('Prob.')
plt.show()


class Train(Suite):
    def __init__(self, hypos, alpha=1.):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, hypo**(-alpha))
        self.Normalize()
    
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1. / hypo

suite = Train(hypos)
observed = 60
suite.Update(observed)
assignments3 = sorted(suite.Items())
hs3, ps3 = [a[0] for a in assignments3], [a[1] for a in assignments3]

plt.plot(hs, ps, 'k-', label='uniform prior')
plt.plot(hs3, ps3, 'r-', label='exp prior')
plt.xlabel('No. Trains')
plt.ylabel('Prob.')
plt.legend()
plt.show()


def Percentile(pmf, percentage):
    p = percentage / 100.
    total = 0
    for val, prob in pmf.Items():
        total += prob
        if total >= p:
            return val

interval = [Percentile(suite, bound) for bound in [5, 95]]
print(interval)


cdf = suite.MakeCdf()
interval = [cdf.Percentile(bound) for bound in [5, 95]]
print(interval)

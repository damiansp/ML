'''
A railroad numbers its locomotives 1...N.  One day you see a locomotive with the
number 60. Estimate how many locomotives the railroad has.
'''
import sys

import matplotlib.pyplot as plt

sys.path.append('..')
from thinkbayes import Suite


# prior
hypos = range(1, 1001)

class Train(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1. / hypo

suite = Train(hypos)
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

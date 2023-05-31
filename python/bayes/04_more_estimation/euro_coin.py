import sys

import matplotlib.pyplot as plt

sys.path.append('..')
import thinkbayes
from thinkbayes import Beta, Suite

class Euro(Suite):
    def Likelihood(self, data, h):
        #x = h
        #if data == 'H':
        #    return x / 100.
        #return 1 - x / 100.
        # Better implementation
        x = h / 100.
        if data == 'H':
            return x
        return 1 - x

suite = Euro(range(101))
dataset = 'H' * 140 + 'T' * 110
for data in dataset:
    suite.Update(data)
    
assignments2 = suite.Items()
hs2, ps2 = [a[0] for a in assignments2], [a[1] for a in assignments2]

plt.plot(hs2, ps2, 'k-', label='first set')
plt.xlabel('P(H)')
plt.ylabel('Prob.')
#plt.show()


def MaximumLikelihood(pmf):
    '''Returns the value with the highest probability'''
    _, val = max((prob, val) for val, prob in pmf.Items())
    return val

print('ML:', MaximumLikelihood(suite))
print('mean:', suite.Mean())
print('median:', thinkbayes.Percentile(suite, 50))
print(f'95CI: {thinkbayes.CredibleInterval(suite, 95)}')


def TrianglePrior():
    suite = Euro()
    for x in range(51):
        suite.Set(x, x)
    for x in range(51, 101):
        suite.Set(x, 100 - x)
    suite.Normalize()
    

# Using Suite.UpdateSet
heads = 150
tails = 100
dataset = 'H' * heads + 'T' * tails
suite.UpdateSet(dataset)

assignments3 = suite.Items()
hs3, ps3 = [a[0] for a in assignments3], [a[1] for a in assignments3]

plt.plot(hs3, ps3, 'r-', label='second set')
plt.legend()
plt.show()



# Using a Beta (conjugate) prior
beta = Beta() # default init is Beta(1, 1) = Unif(0, 1)
beta.Update((140, 110))
print(beta.Mean())
beta.Update((150, 100))
print(beta.Mean())

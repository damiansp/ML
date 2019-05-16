import sys

sys.path.append('..')
from thinkbayes import Pmf


class Monty(Pmf):
    def __init__(self, hypos):
        Pmf.__init__(self)
        for h in hypos:
            self.Set(h, 1)
        self.Normalize()

    def Update(self, data):
        for h in self.Values():
            like = self.Likelihood(data, h)
            self.Mult(h, like)
        self.Normalize()

    def Likelihood(self, data, h):
        if h == data:
            return 0
        if h == 'A':
            return 0.5
        return 1


Hs = 'ABC'
pmf = Monty(Hs)
data = 'B'
pmf.Update(data)
for h, p in pmf.Items():
    print(f'{h}: {p}')

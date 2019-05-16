import sys

sys.path.append('..')
from thinkbayes import Suite


class Monty(Suite):
    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        if hypo == 'A':
            return 0.5
        return 1


suite = Monty('ABC')
data = 'B'
suite.Update(data)
suite.Print()

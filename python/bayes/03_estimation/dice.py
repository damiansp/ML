import sys

sys.path.append('..')
from thinkbayes import Suite


class Dice(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        return 1. / hypo


suite = Dice([4, 6, 8, 12, 20])
data = 6
suite.Update(data)
suite.Print()

more_data = [6, 8, 7, 7, 6, 4]
for observation in more_data:
    suite.Update(observation)
suite.Print()

import sys

sys.path.append('..')
from thinkbayes import Suite


class MAndM(Suite):
    mix94 = {'brown': 30,
             'yellow': 20,
             'red': 20,
             'green': 10,
             'orange': 10,
             'tan': 10}
    mix96 = {'blue': 24,
             'green': 20,
             'orange': 16,
             'yellow': 14,
             'red': 13,
             'brown': 13}
    hA = {'bag1': mix94, 'bag2': mix96}
    hB = {'bag1': mix96, 'bag2': mix94}
    Hs = {'A': hA, 'B': hB}
    
    def Likelihood(self, data, h):
        bag, color = data
        mix = self.Hs[h][bag]
        like = mix[color]
        return like


suite = MAndM('AB')
suite.Update(('bag1', 'yellow'))
suite.Update(('bag2', 'green'))
suite.Print()

import sys

import matplotlib.pyplot as plt

sys.path.append('..')
from thinkbayes import Pmf, SampleSum


class Die(Pmf):
    def __init__(self, sides):
        Pmf.__init__(self)
        for x in range(1, sides + 1):
            self.Set(x, 1)
        self.Normalize()

d6 = Die(6)
dice = [d6] * 3
three = SampleSum(dice, 1000) # simulated
sim = sorted(three.Items())
h_sim, p_sim = [a[0] for a in sim], [a[1] for a in sim]
plt.plot(h_sim, p_sim, 'k-', label='simulated')
plt.xlabel('P(H)')
plt.ylabel('Prob.')

three_exact = d6 + d6 + d6
exact = sorted(three_exact.Items())
h_ex, p_ex = [a[0] for a in exact], [a[1] for a in exact]
plt.plot(h_ex, p_ex, 'r-', label='exact')
plt.legend()
plt.show()

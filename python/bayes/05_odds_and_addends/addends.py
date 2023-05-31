import sys

import matplotlib.pyplot as plt

sys.path.append('..')
from thinkbayes import MakeMixture, Pmf, SampleSum


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


# Maxima
def RandomMax(dists):
    total = max(dist.Random() for dist in dists)
    return total


def SampleMax(dists, n):
    pmf = MakePmfFromList(RandomMax(dists) for _ in range(n))
    return pmf


def PmfMax(pmf1, pmf2):
    res = Pmf()
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            res.Incr(max(v1, v2), p1 * p2)
    return res

# Far more efficient:
# In class Cdf:
#def Max(self, k):
#    cdf = self.Copy()
#    cdf.ps = [p ** k for p in cdf.ps]
#    return cdf

best_attr_cdf = three_exact.Max(6)
best_attr_pmf = best_attr_cdf.MakePmf()
max_items = sorted(best_attr_pmf.Items())
h_max, p_max = [a[0] for a in max_items], [a[1] for a in max_items]
plt.plot(h_max, p_max)
plt.xlabel('Max value')
plt.ylabel('Prob')
plt.show()


# Randomly choose 1 die from the set and roll. What is P(outcome)?
# 2 dice example: 1 6-sided, 1 8-sided
d6 = Die(6)
d8 = Die(8)
mix = Pmf()
for die in [d6, d8]:
    for outcome, prob in die.Items():
        mix.Incr(outcome, prob)
mix.Normalize()


# More complicated set of dice
pmf_dice = Pmf()
pmf_dice.Set(Die(4), 2) # 2 4-sided dice
pmf_dice.Set(Die(6), 3) # 3 6-sided, etc
pmf_dice.Set(Die(8), 2)
pmf_dice.Set(Die(12), 1)
pmf_dice.Set(Die(20), 1)
pmf_dice.Normalize()

mix = MakeMixture(pmf_dice)
mix_items = sorted(mix.Items())
h_mix, p_mix = [a[0] for a in mix_items], [a[1] for a in mix_items]
plt.plot(h_mix, p_mix)
plt.xlabel('Outcome')
plt.ylabel('Prob')
plt.show()

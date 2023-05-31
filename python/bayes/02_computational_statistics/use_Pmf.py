import sys

sys.path.append('..')
from thinkbayes import Pmf


pmf = Pmf()
for x in range(1, 7):
    pmf.Set(x, 1./6)


# another example:
# pmf = Pmf()
# for word in word_list:
#     pmf.Incr(word, 1)
# pmf.Normalize()
# print(pmf.Prob('the'))


# The cookie problem
pmf = Pmf()
pmf.Set('Bowl1', 0.5) # P(B1)
pmf.Set('Bowl2', 0.5) # P(B2)

pmf.Mult('Bowl1', 0.75) # P(V|B1)
pmf.Mult('Bowl2', 0.5)  # P(V|B2)
pmf.Normalize()

# P(B1|V)
print(pmf.Prob('Bowl1'))



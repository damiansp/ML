import matplotlib.pyplot as plt
from pygam import LinearGAM
from pygam.datasets import wage


X, y = wage(return_X_y=True)
gam = LinearGAM(n_splines=10).gridsearch(X, y)
XX = gam.generate_X_grid
fig, axs = plt.subplots(1, 3)
titles = ['year', 'age', 'education']

for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i, width=0.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], *confi, c='r')
    ax.set_title(titles[i])

plt.show()

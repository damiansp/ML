import pyro
import pyro.distributions as dist
import torch

# Primitive stochastic functions
# Draw sample from standard normal distribution
loc = 0
scale = 1
normal = dist.Normal(loc, scale)
x = normal.sample()
print('sample:', x)
print('log prob:', normal.log_prob(x))

x = pyro.sample('my_sample', dist.Normal(loc, scale))
print(x)



# A Simple Model
def weather():
    cloudy = pyro.sample('cloudy', dist.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55., 'sunny': 75.}[cloudy]
    scale_temp = {'cloudy': 10., 'sunny': 15.}[cloudy]
    temp = pyro.sample('temp', dist.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

for _ in range(3):
    print(weather())


def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80. else 50.
    ice_cream = pyro.sample('ice_cream', dist.Normal(expected_sales, 10.))
    return ice_cream


def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample('x_{}'.format(t), dist.Bernoulli(p))
    if x.item() == 0:
        return x
    return x + geometric(p, t + 1)

print(geometric(0.5))


def normal_product(loc, scale):
    z1 = pyro.sample('z1', dist.Normal(loc, scale))
    z2 = pyro.sample('z2', dist.Normal(loc, scale))
    y = z1 * z2
    return y


def make_normal_normal():
    mu_latent = pyro.sample('mu_latent', dist.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn

print(make_normal_normal()(1.))


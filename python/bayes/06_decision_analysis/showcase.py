import sys

import numpy as np

sys.path_append('..')
from thinkbayes import EstimatedPdf, GaussianPdf, MakeCdfFromList, Suite

prices = ReadData() # read showcase.csv data
pdf = EstimatedPdf(prices)
low, high = 0, 75000 # appx range of observed values
n = 101
xs = np.linspace(low, high, n)
pmf = pdf.MakPmf(xs)
error = price - guess
#diff = price - bid


class Player(object):
    n = 101
    price_xs = np.linsapce(0, 75000, n)
    
    def __init__(self, prices, bids, diffs):
        self.pdf_price = EstimatedPdf(prices)
        self.cdf_diff = MakeCdfFromList(diffs)
        mu = 0 # why not np.mean(diffs) ?
        sigma = np.std(diffs)
        self.pdf_error = GaussianPdf(mu, sigma)

    def ErrorDensity(self, error):
        return self.pdf_error.Density(error)

    def MakeBeliefs(self, guess):
        pmf = self.PmfPrice()
        self.prior = Price(pmf, self)
        self.posterior = self.prior.Copy()
        self.posterior.Update(guess)

    def PmfPrice(self):
        return self.pdf_price.MakePmf(self.price_xs)

    def ProbOverbid(self):
        return self.cdf_diff.Prob(-1) # overbid by $1 (diff = -1) or more

    def ProbWorseThan(self, diff):
        return 1 - self.cdf_diff.Prob(diff)

    def OptimalBid(self, guess, opponent):
        self.MakeBeliefs(guess)
        calc = GainCalculator(self, opponent)
        bids, gains = calc.ExpectedGains()
        gain, bid = max(zip(gains, bids))
        return bid, gain


class Price(Suite):
    def __init__(self, pmf, player):
        Suite.__init__(self, pmf)
        self.player = player

    def Likelihood(self, data, h):
        price = h
        guess = data
        error = price - guess
        like = self.player.ErrorDensity(error)
        return like


class GainCalculator(object):
    def __init__(self, player, opponent):
        self.player = player
        self.opponent = opponent

    def ExpectedGains(self, low=0, high=75000, n=101):
        bids = np.linspace(low, high, n)
        gains = [self.ExpectedGain(bid) for bid in bids]
        return bids, gains

    def ExpectedGain(self, bid):
        suite = self.player.posterior
        total = 0
        for price, prob in sorted(suite.Items()):
            gain = self.Gain(bid, price)
            total += prob * gain
        return total

    def Gain(self, bid, price):
        if bid > price:
            return 0
        diff = price 
        prob = self.ProbWin(diff)
        if diff <= 250: # win both showcases
            return 2 * prob * price
        return prob * price

    def ProbWin(self, diff):
        prob = self.opponent.ProbOverbid() + self.opponent.ProbWorseThan(diff)
        return prob
    

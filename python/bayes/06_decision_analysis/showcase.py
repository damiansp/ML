prices = ReadData() # read showcase.csv data
pdf = thinkbayes.EstimatedPdf(prices)
low, high = 0, 75000 # appx range of observed values
n = 101
xs = np.linspace(low, high, n)
pmf = pdf.MakPmf(xs)
error = price - guess
                 

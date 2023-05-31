def Odds(p):
    return p / (1 - p)

def Prob(odds):
    return odds / (odds + 1)

def ProbFrac(yes, no):
    return yes / (yes + no)

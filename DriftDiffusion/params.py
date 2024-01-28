import matplotlib.pyplot as plt

def pretty(state: bool = True):
    """ plotting params update for LaTex style graphs """
    if state is True:
        plt.rcParams.update({'text.usetex': True, # enable LaTex formatting
                     'grid.linestyle':'dotted', # dotted grid background
                     'figure.figsize': (7, 5), # figure size (width, height) in inches
                     'font.family': 'Computer Modern Roman'}) # font (default LaTex font)
        
    else:
        pass
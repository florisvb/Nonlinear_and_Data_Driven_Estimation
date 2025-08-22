import matplotlib
font = {'size': 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

def plot_tme(t, true, measured, estimated=None, ax=None, label_var='y'):
    if ax is None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)

    if measured is not None:
        ax.plot(t, measured, '*', color='blue', label=label_var + ' measured', markersize=0.5)
        
    if estimated is not None:
        ax.plot(t, estimated, color='red', label=label_var + ' hat')
        
    if true is not None:
        ax.plot(t, true, '--', color='black', label=label_var + ' true')

    ax.set_xlabel('Time')
    ax.set_ylabel(label_var)
    
    ax.legend()
    
    return ax
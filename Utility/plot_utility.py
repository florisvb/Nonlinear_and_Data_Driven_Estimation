import matplotlib
font = {'size': 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

def plot_tme(t, true, measured, estimated=None, ax=None, label_var='y', markersize=2):
    if ax is None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)

    if measured is not None:
        ax.plot(t, measured, '*', color='blue', label=label_var + ' measured', markersize=markersize)
        
    if estimated is not None:
        ax.plot(t, estimated, color='red', label=label_var + ' hat')
        
    if true is not None:
        ax.plot(t, true, '--', color='black', label=label_var + ' true')

    ax.set_xlabel('Time')
    ax.set_ylabel(label_var)
    
    ax.legend()
    
    return ax

def plot_planar_drone(df_trajec, axes=None):
    if axes is None:
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        ax1, ax2 = axes
        
    ax1.plot(df_trajec.time.values, df_trajec.theta.values)
    ax1.set_xlabel('Time, sec')
    ax1.set_ylabel('theta')
    
    ax2.plot(df_trajec.x.values, df_trajec.z.values, '.')
    ax2.set_xlabel('x pos')
    ax2.set_ylabel('z pos')

    return ax1, ax2

# From claude
def find_contiguous_chunks(lst):
    """
    Find contiguous chunks in a list of integers.

    Parameters:
    -----------
    lst : list of int
        List of integers

    Returns:
    --------
    list of lists : Each sublist contains a contiguous sequence
    """
    if not lst:
        return []

    lst = sorted(lst)
    chunks = []
    current_chunk = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1:
            current_chunk.append(lst[i])
        else:
            chunks.append(current_chunk)
            current_chunk = [lst[i]]

    chunks.append(current_chunk)
    return chunks

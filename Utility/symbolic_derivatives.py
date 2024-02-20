import sympy as sp
import numpy as np

def directional_derivative(h, x, f):
    '''
    Essentially does the chain rule, but replaces xdot with f.
    '''
    if type(x) == list:
        x = np.matrix(x).T
    if type(f) == list:
        f = np.matrix(f).T
        
    n = len(x) # number of states
    try:
        _ = h.shape
    except:
        h = sp.Matrix([[h]])
    hdx = sp.zeros(h.shape[0], n)
    for i in range(hdx.shape[0]):
        for j in range(n):
            hdx[i,j] = sp.diff(h[i,0], x[j,0]) #dx[j,0]
            
    #print(hdx)
    return hdx*f  # multiplying by f is just like multiplying by the dx, e.g. the chain rule

def chain_rule_derivative(h, x, dx):
    return directional_derivative(h, x, dx)

def get_bigO(h, x, fs):
    O = [h]
    for f in fs:
        h_f = directional_derivative(h, x, f)
        O.append(h_f)
    return O


def get_vars(exprs):
    '''
    Get all the free variables from a list of expressions
    '''
    dx = []
    try:
        exprs = sp.Matrix.vstack(*exprs)
    except:
        pass # hopefully exprs is a matrix
    for expr in exprs:
        free_symbols = expr.free_symbols
        for sym in free_symbols:
            if sym not in dx:
                dx.append(sym)
    return dx
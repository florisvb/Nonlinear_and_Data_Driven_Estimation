import sympy as sp
import numpy as np

# capture things like sin = sp.sin
from sympy import *

import json
import inspect

# Build fast lambda functions: mostly written by Claude
def build_lambda_funcs(feature_names, symbols):
    # replace ' ' with multiplication
    for i, feature in enumerate(feature_names):
      feature = feature.replace(' ', '*')
      feature_names[i] = feature
    
    # Parse feature names into SymPy expressions
    sympy_exprs = []
    for name in feature_names:
          expr = sp.sympify(name.replace('^', '**').replace(' ', '*'))
          sympy_exprs.append(expr)

    # Convert to fast lambda functions
    lambda_funcs = [sp.lambdify(symbols, expr, 'numpy') for expr in sympy_exprs]

    return lambda_funcs 

# When saving: mostly written by Claude
def save_sindy_portable(lambda_funcs, coefficients, filename):
    func_strings = []
    for i, func in enumerate(lambda_funcs):
        try:
            source = inspect.getsource(func).strip()
        except:
            import dill
            source = dill.source.getsource(func).strip()
        func_strings.append(source)
    
    data = {
        'function_strings': func_strings,
        'coefficients': coefficients.tolist() if hasattr(coefficients, 'tolist') else coefficients
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# When loading: mostly written by Claude
def load_sindy_portable(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Create comprehensive namespace with all common math functions
    namespace = {
        'np': np,
        'numpy': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'arctan2': np.arctan2,
        'pi': np.pi,
        'e': np.e,
    }
    
    # Recreate functions
    lambda_funcs = []
    for func_str in data['function_strings']:
        # Use exec for function definitions
        exec(func_str, namespace)
        
        # Extract the function from namespace
        if func_str.strip().startswith('lambda'):
            # It's a lambda expression
            lambda_funcs.append(eval(func_str, namespace))
        else:
            # It's a def function - extract function name
            func_name = func_str.split('(')[0].replace('def', '').strip()
            lambda_funcs.append(namespace[func_name])
    
    coefficients = np.array(data['coefficients'])
    return lambda_funcs, coefficients

# Usage:
#save_lambdas_portable(lambda_funcs, coefficients, 'sindy_model.json')
#lambda_funcs, coefficients = load_lambdas_portable('sindy_model.json')

# Assemble an inference model
class SINDY_INFERENCE_MODEL:
    def __init__(self, lambda_funcs, coefficients):
        self.lambda_funcs = lambda_funcs
        self.coefficients = coefficients

    def predict(self, x, u=None):
        if u is not None:
            cols = np.hstack([np.ravel(x), np.ravel(u)])
        else:
            cols = np.ravel(x)

        results = np.array([func(*cols) for func in self.lambda_funcs])
        v = self.coefficients@np.atleast_2d(results).T
        return v


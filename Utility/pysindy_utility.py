import dill
import sympy as sp

# capture things like sin = sp.sin
from sympy import *

    
def build_lambda_funcs(feature_names, symbols):
    # replace ' ' with multiplication
    for i, feature in enumerate(feature_names):
      feature = feature.replace(' ', '*')
      feature_names[i] = feature
    
    # Parse feature names into SymPy expressions
    sympy_exprs = []
    for name in feature_names:
          expr = sp.sympify(name.replace('^', '**'))
          sympy_exprs.append(expr)

    # Convert to fast lambda functions
    lambda_funcs = [sp.lambdify(symbols, expr, 'numpy') for expr in sympy_exprs]

    return lambda_funcs 

class SINDY_INFERENCE_MODEL:
    def __init__(self, lambda_funcs, coefficients):
        self.lambda_funcs = lambda_funcs
        self.coefficients = coefficients

    def predict(self, x, u):
        input = np.hstack([np.ravel(x), np.ravel(u)])
        results = np.array([func(*cols) for func in self.lambda_funcs])
        v = self.coefficients@np.atleast_2d(results).T
        return v

def save_sindy_lambda_funcs_and_coefficients(lambda_funcs, coefficients, filename):
    sindy_model = {'lambda_funcs': lambda_funcs,
                   'coefficients': coefficients}

    with open(filename, 'wb') as f:
        dill.dump(sindy_model, f)

def load_sindy_lambda_funcs_and_coefficients(filename):
    with open(filename, 'rb') as f:
        loaded_data = dill.load(f)
    return loaded_data['lambda_funcs'], loaded_data['coefficients']


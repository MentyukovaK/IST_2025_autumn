import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.optimize import line_search
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
from collections import defaultdict
import warnings

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func(), .grad() methods implemented for computing
            function values and its gradient.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c
        
        # Get initial alpha
        alpha = previous_alpha if previous_alpha is not None else self.alpha_0
        phi_0 = oracle.func(x_k)
        grad_0 = oracle.grad(x_k)
        phi_der_0 = np.dot(grad_0, d_k)
        
        # If gradient directional is non-negative, direction is not descent
        if phi_der_0 >= 0:
            return None
        
        if self._method == 'Wolfe':
            try:
                # Use scipy.optimize.line_search for Wolfe conditions
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = line_search(
                        f=lambda x: oracle.func(x),
                        myfprime=lambda x: oracle.grad(x),
                        xk=x_k,
                        pk=d_k,
                        gfk=grad_0,
                        old_fval=phi_0,
                        old_old_fval=None,
                        c1=self.c1,
                        c2=self.c2,
                        amax=100
                    )
                
                # scipy.optimize.line_search returns a tuple where the first element is alpha
                alpha_wolfe = result[0]
                
                if alpha_wolfe is not None and alpha_wolfe > 0:
                    return alpha_wolfe
            except Exception as e:
                # If Wolfe fails, we'll fall back to Armijo
                pass
        
        # Armijo rule (or fallback from Wolfe)
        rho = 0.5  # shrinkage factor for backtracking
        alpha_curr = alpha
        
        while True:
            phi_alpha = oracle.func(x_k + alpha_curr * d_k)
            if phi_alpha <= phi_0 + self.c1 * alpha_curr * phi_der_0:
                return alpha_curr
            alpha_curr *= rho
            if alpha_curr < 1e-12:  # too small step
                return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() methods implemented for computing
        function value and its gradient.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    
    for iter_num in range(max_iter):
        # Compute function value and gradient
        try:
            f_k = oracle.func(x_k)
            grad_k = oracle.grad(x_k)
        except Exception as e:
            if display:
                print(f"Computational error at iteration {iter_num}: {str(e)}")
            return x_k, 'computational_error', history
        
        # Check for computational errors
        if np.isnan(f_k) or np.isnan(grad_k).any() or np.isinf(f_k) or np.isinf(grad_k).any():
            if display:
                print(f"NaN or Inf detected at iteration {iter_num}")
            return x_k, 'computational_error', history
        
        # Compute gradient norm
        grad_norm = np.linalg.norm(grad_k)
        
        # Save to history
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(f_k)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        # Stopping criterion
        if grad_norm < tolerance:
            if display:
                print(f"Converged in {iter_num} iterations")
            return x_k, 'success', history
        
        # Search direction is negative gradient
        d_k = -grad_k
        
        # Line search
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
        
        # Check if line search failed
        if alpha_k is None or alpha_k <= 0:
            if display:
                print(f"Line search failed at iteration {iter_num}")
            return x_k, 'computational_error', history
        
        # Update point
        x_new = x_k + alpha_k * d_k
        
        # Check for computational errors after update
        if np.isnan(x_new).any() or np.isinf(x_new).any():
            if display:
                print(f"NaN or Inf after update at iteration {iter_num}")
            return x_k, 'computational_error', history
        
        x_k = x_new
    
    # Max iterations exceeded
    if display:
        print(f"Maximum iterations ({max_iter}) exceeded")
    
    # Final evaluation for history
    if trace:
        try:
            f_k = oracle.func(x_k)
            grad_k = oracle.grad(x_k)
            grad_norm = np.linalg.norm(grad_k)
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(f_k)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        except:
            pass
    
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    
    for iter_num in range(max_iter):
        # Compute function value, gradient and Hessian
        try:
            f_k = oracle.func(x_k)
            grad_k = oracle.grad(x_k)
            hess_k = oracle.hess(x_k)
        except Exception as e:
            if display:
                print(f"Computational error at iteration {iter_num}: {str(e)}")
            return x_k, 'computational_error', history
        
        # Check for computational errors
        if np.isnan(f_k) or np.isnan(grad_k).any() or np.isinf(f_k) or np.isinf(grad_k).any():
            if display:
                print(f"NaN or Inf detected at iteration {iter_num}")
            return x_k, 'computational_error', history
        
        if np.isnan(hess_k).any() or np.isinf(hess_k).any():
            if display:
                print(f"NaN or Inf in Hessian at iteration {iter_num}")
            return x_k, 'computational_error', history
        
        # Compute gradient norm
        grad_norm = np.linalg.norm(grad_k)
        
        # Save to history
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(f_k)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        # Stopping criterion
        if grad_norm < tolerance:
            if display:
                print(f"Converged in {iter_num} iterations")
            return x_k, 'success', history
        
        try:
            # Use Cholesky decomposition for solving Newton system
            # For non-symmetric matrices, use the symmetric part
            hess_k_sym = (hess_k + hess_k.T) / 2
            
            # Add small regularization to ensure positive definiteness
            eps = 1e-8
            if hess_k_sym.shape[0] > 0:
                eps = 1e-8 * np.trace(hess_k_sym) / hess_k_sym.shape[0]
            
            # Try Cholesky decomposition
            c, low = cho_factor(hess_k_sym + eps * np.eye(hess_k_sym.shape[0]))
            d_k = cho_solve((c, low), -grad_k)
            
            # Check if direction is descent direction
            if np.dot(grad_k, d_k) >= 0:
                if display:
                    print(f"Newton direction is not a descent direction at iteration {iter_num}")
                return x_k, 'newton_direction_error', history
                
        except (LinAlgError, ValueError) as e:
            if display:
                print(f"Cholesky decomposition failed at iteration {iter_num}: {str(e)}")
            return x_k, 'newton_direction_error', history
        
        # Line search
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
        
        # Check if line search failed
        if alpha_k is None or alpha_k <= 0:
            if display:
                print(f"Line search failed at iteration {iter_num}")
            return x_k, 'computational_error', history
        
        # Update point
        x_new = x_k + alpha_k * d_k
        
        # Check for computational errors after update
        if np.isnan(x_new).any() or np.isinf(x_new).any():
            if display:
                print(f"NaN or Inf after update at iteration {iter_num}")
            return x_k, 'computational_error', history
        
        x_k = x_new
    
    # Max iterations exceeded
    if display:
        print(f"Maximum iterations ({max_iter}) exceeded")
    
    # Final evaluation for history
    if trace:
        try:
            f_k = oracle.func(x_k)
            grad_k = oracle.grad(x_k)
            grad_norm = np.linalg.norm(grad_k)
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(f_k)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        except:
            pass
    
    return x_k, 'iterations_exceeded', history
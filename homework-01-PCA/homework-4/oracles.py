import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = b.size

    def func(self, x):
        # Compute Ax for all samples
        Ax = self.matvec_Ax(x)
        
        # Compute -b * (Ax) element-wise
        bAx = -self.b * Ax
        
        # Use logaddexp for numerical stability: log(1 + exp(z)) = logaddexp(0, z)
        log_terms = np.logaddexp(0, bAx)
        
        # Mean of log terms + L2 regularization
        return np.mean(log_terms) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        # Compute Ax for all samples
        Ax = self.matvec_Ax(x)
        
        # Compute -b * (Ax) element-wise
        bAx = -self.b * Ax
        
        # Compute sigma(-b * (Ax)) using expit for numerical stability
        sigma = expit(bAx)
        
        # Compute gradient: -1/m * A^T (b * sigma) + regcoef * x
        grad_part = self.matvec_ATx(self.b * sigma)
        return -grad_part / self.m + self.regcoef * x

    def hess(self, x):
        # Compute Ax for all samples
        Ax = self.matvec_Ax(x)
        
        # Compute -b * (Ax) element-wise
        bAx = -self.b * Ax
        
        # Compute p = expit(-b * (Ax))
        p = expit(bAx)
        
        # Compute weights for the diagonal matrix: p * (1 - p)
        weights = p * (1 - p)
        
        # Compute Hessian: 1/m * A^T * diag(weights) * A + regcoef * I
        hess_part = self.matmat_ATsA(weights)
        return hess_part / self.m + self.regcoef * np.eye(x.size)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._Ax_cache = None
        self._Ad_cache = None

    def _update_caches(self, x, d):
        """Update cached values for Ax and Ad if needed"""
        if self._Ax_cache is None:
            self._Ax_cache = self.matvec_Ax(x)
        if self._Ad_cache is None:
            self._Ad_cache = self.matvec_Ax(d)

    def func_directional(self, x, d, alpha):
        # Precompute Ax and Ad if not cached
        if self._Ax_cache is None or self._Ad_cache is None:
            self._Ax_cache = self.matvec_Ax(x)
            self._Ad_cache = self.matvec_Ax(d)
        
        # Compute A(x + alpha*d) = Ax + alpha*Ad
        Ax_alpha = self._Ax_cache + alpha * self._Ad_cache
        
        # Compute -b * (A(x + alpha*d)) element-wise
        bAx_alpha = -self.b * Ax_alpha
        
        # Use logaddexp for numerical stability
        log_terms = np.logaddexp(0, bAx_alpha)
        
        # Mean of log terms + L2 regularization
        x_alpha = x + alpha * d
        reg_term = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        
        return np.mean(log_terms) + reg_term

    def grad_directional(self, x, d, alpha):
        # Precompute Ax and Ad if not cached
        if self._Ax_cache is None or self._Ad_cache is None:
            self._Ax_cache = self.matvec_Ax(x)
            self._Ad_cache = self.matvec_Ax(d)
        
        # Compute A(x + alpha*d) = Ax + alpha*Ad
        Ax_alpha = self._Ax_cache + alpha * self._Ad_cache
        
        # Compute -b * (A(x + alpha*d)) element-wise
        bAx_alpha = -self.b * Ax_alpha
        
        # Compute sigma(-b * (A(x + alpha*d))) using expit
        sigma = expit(bAx_alpha)
        
        # Compute gradient of f(x + alpha*d) in direction d
        # = d^T * grad f(x + alpha*d)
        # = d^T * (-1/m * A^T (b * sigma) + regcoef * (x + alpha*d))
        
        # First part: d^T * (-1/m * A^T (b * sigma))
        # = -1/m * (A*d)^T * (b * sigma)
        Ad = self._Ad_cache
        first_part = -np.dot(Ad, self.b * sigma) / self.m
        
        # Second part: d^T * (regcoef * (x + alpha*d))
        x_alpha = x + alpha * d
        second_part = self.regcoef * np.dot(d, x_alpha)
        
        return first_part + second_part

    def _clear_cache(self):
        """Clear cached values"""
        self._Ax_cache = None
        self._Ad_cache = None

    def func(self, x):
        self._clear_cache()
        return super().func(x)

    def grad(self, x):
        self._clear_cache()
        return super().grad(x)

    def hess(self, x):
        self._clear_cache()
        return super().hess(x)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    # Define matrix-vector product functions
    def matvec_Ax(x):
        return A.dot(x)
    
    def matvec_ATx(x):
        return A.T.dot(x)
    
    def matmat_ATsA(s):
        # Handle both dense and sparse matrices
        if scipy.sparse.issparse(A):
            # For sparse matrices: A.T @ diag(s) @ A
            if hasattr(s, "flatten"):
                s = s.flatten()
            # Create a diagonal sparse matrix from s
            S = scipy.sparse.diags(s)
            result = A.T.dot(S.dot(A))
            # Convert to dense if the result is small, keep sparse otherwise
            if result.shape[0] <= 1000:  # Threshold for converting to dense
                return result.toarray()
            return result
        else:
            # For dense matrices: use broadcasting for efficiency
            return A.T.dot(A * s[:, np.newaxis])

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = x.size
    grad = np.zeros(n)
    f_x = func(x)
    
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        f_x_eps = func(x_eps)
        grad[i] = (f_x_eps - f_x) / eps
    
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = x.size
    hess = np.zeros((n, n))
    f_x = func(x)
    
    # Precompute f(x + eps * e_i) for all i
    f_x_eps_i = np.zeros(n)
    for i in range(n):
        x_eps_i = x.copy()
        x_eps_i[i] += eps
        f_x_eps_i[i] = func(x_eps_i)
    
    # Compute Hessian entries
    for i in range(n):
        for j in range(i, n):  # Use symmetry
            x_eps_ij = x.copy()
            x_eps_ij[i] += eps
            x_eps_ij[j] += eps
            f_x_eps_ij = func(x_eps_ij)
            
            hess_ij = (f_x_eps_ij - f_x_eps_i[i] - f_x_eps_i[j] + f_x) / (eps ** 2)
            hess[i, j] = hess_ij
            if i != j:
                hess[j, i] = hess_ij  # Symmetry
    
    return hess
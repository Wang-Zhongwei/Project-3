import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp


class Diffusion:
    _A: np.ndarray

    def __init__(self, domain, d, ic, bc, src) -> None:
        """

        Args:
            domain (tuple): tuple of domain for each dimensions
            d (float): grid spacing
            bc (str): boundary conditions (dirichlet, neumann, robin or periodic)
            ic (callable): initial conditions
            src (callable): heat source
        """
        self._d = d
        # check boundary conditions
        if bc == 'dirichlet':
            # create grid of interior points
            grids_1d = [np.linspace(dom[0] + d, dom[1] - d, int(
                (dom[1] - dom[0])/d) - 1) for dom in domain]
            self._grids = np.meshgrid(*grids_1d, indexing='ij')
        elif bc in ('neumann', 'robin', 'periodic'):
            raise NotImplementedError(
                f'Boundary conditions{bc} not implemented')
        else:
            raise ValueError(f'Boundary conditions{bc} not supported')
        self._bc = bc

        ic = np.vectorize(ic)
        self._ic = ic(*self._grids)

        self._src = np.vectorize(src)

    def solve(self, t_span, h, method='euler'):
        """time evolution

        Args:
            t_span (tuple): (t_initial, t_final)
            h (float): time step size
            method (str, optional): Method. Defaults to 'euler'. 
                Supported methods: 
                - euler: Forward Euler
                - trapezoid: Trapezoidal rule
                - MoL: Method of Lines
        """
        self._h = h
        mu = h/self._d**2
        self.t_eval = np.linspace(t_span[0], t_span[1], int(
            (t_span[1] - t_span[0])/h) + 1)

        if (method == 'MoL'):
            if not isinstance(self._A, np.ndarray):
                A = self._A.toarray()
            else:
                A = self._A
            sol = solve_ivp(fun=lambda t, u: u + A / self._d**2 @ u.T +
                            self._src(t, *self._grids).flatten(order='f'),
                            y0=self._ic.flatten(order='f'),
                            t_span=t_span, method='RK45',
                            t_eval=self.t_eval)
            self.solution = sol.y.T.reshape(
                (len(self.t_eval), *self._ic.shape), order='f')
        else:
            self.solution = [self._ic]
            self._load_operator(mu, method)
            old = self._ic
            for i in range(len(self.t_eval) - 1):
                t = self.t_eval[i]
                if method == 'euler':
                    new = self._euler_step(old, t, h)
                elif method == 'trapezoid':
                    new = self._trapezoid_step(old, t, h)
                else:
                    raise ValueError(f'Unknown method {method}')
                self.solution.append(new)
                old = new

        return self.t_eval, self.solution

    def _euler_step(self, old, t, h):
        src_term = h * self._src(t, *self._grids)
        new = (self._evo_op @ old.flatten(order='f').T).reshape(old.shape,
                                                                order='f') + src_term
        return new

    def _trapezoid_step(self, old, t, h):
        src_term = h/2 * (self._half_evo_op @ (self._src(t, *self._grids) +
                          self._src(t + h, *self._grids)).flatten(order='f').T).reshape(old.shape, order='f')
        new = (self._evo_op @ old.flatten(order='f').T).reshape(old.shape,
                                                                order='f') + src_term
        return new

    def _load_operator(self, mu, method):
        I = sp.eye(self._A.shape[0], format='csr')
        if method == 'euler':
            self._evo_op = I + mu * self._A
        elif method == 'trapezoid':
            if not isinstance(self._A, np.ndarray):
                # convert to dense matrix so that inverse can be calculated using np.linalg.inv
                dense_A = self._A.toarray()
            self._half_evo_op = np.linalg.inv(I - mu * dense_A/2)
            self._evo_op = self._half_evo_op @ (I + mu * self._A/2)
        else:
            raise ValueError(f'Unknown method {method}')

    def calc_error(self, expected_func):
        expected_sol = [np.vectorize(expected_func)(
            t, *self._grids) for t in self.t_eval]
        return [self._d * np.linalg.norm(sol - exp, ord='fro') for sol, exp in zip(self.solution, expected_sol)]


class Diffusion2D(Diffusion):
    def __init__(self, domain, d, ic, bc, src) -> None:
        super().__init__(domain, d, ic, bc, src)
        self._A = self._get_A()

    def _get_A(self):
        """2D second-order finite difference matrix

        Returns:
            A: sparse matrix representation of the 2D second-order finite difference matrix
        """
        m, n = self._ic.shape
        # create A0 block
        A0 = sp.spdiags([np.ones(m), -4. * np.ones(m),
                        np.ones(m)], [-1., 0., 1.], m, m, format='lil')

        # set diagonal blocks as A0
        A = sp.block_diag([A0] * n, format='lil')
        # set off-diagonal blocks as I
        A.setdiag(1, m)
        A.setdiag(1, -m)

        return A.tocsr()

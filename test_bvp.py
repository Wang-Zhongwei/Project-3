import unittest
from bvp import *
import matplotlib.pyplot as plt

class TestDiffusion(unittest.TestCase):
    def setUp(self) -> None:
        # test heat source
        self.test_src = lambda t, x, y: 2*t*(1 + np.pi**2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

        # test initial condition
        self.test_ic = lambda x, y: 0

        # expected solution
        self.u = lambda t, x, y: t**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def test_init(self):
        solver = Diffusion2D(domain=((0, 1), (0, 2)), d=0.5, ic=self.test_ic, bc='dirichlet', src=0)
        print(solver._ic)
        print(solver._A)
    
    def test_euler(self):
        solver = Diffusion2D(domain=((0, 1), (0, 1)), d=0.1, ic=self.test_ic, bc='dirichlet', src=self.test_src)

        solver.solve(t_span=(0, 1), h=0.05, method='euler')

        plt.imshow(solver.solution[-1].T, cmap='hot', interpolation='bicubic', origin='lower')
        plt.show()
    
    def test_trapezoid(self):
        solver = Diffusion2D(domain=((0, 1), (0, 1)), d=0.1, ic=self.test_ic, bc='dirichlet', src=self.test_src)

        solver.solve(t_span=(0, 1), h=1, method='trapezoid')

        plt.imshow(solver.solution[-1].T, cmap='hot', interpolation='none', origin='lower')
        plt.show()

    def test_MoL(self):
        solver = Diffusion2D(domain=((0, 1), (0, 1)), d=0.1, ic=self.test_ic, bc='dirichlet', src=self.test_src)

        _, solution = solver.solve(t_span=(0, 1), h=0.1, method='MoL')

        plt.imshow(solution[-1].T, cmap='hot', interpolation='bicubic', origin='lower')
        plt.show()

    def test_expected(self):
        d = .1
        x, y = np.linspace(d, 1 - d, 9), np.linspace(d, 1 - d, 9)
        x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')

        plt.imshow(self.u(1, x_mesh, y_mesh).T, cmap='hot', interpolation='bicubic')
        plt.show()

        # numerical solution
        solver = Diffusion2D(domain=((0, 1), (0, 1)), d=0.1, ic=self.test_ic, bc='dirichlet', src=self.test_src)
        solver.solve(t_span=(0, 1), h=0.05, method='MoL')

        # analytical solution
        x, y = np.linspace(d, 1 - d, 9), np.linspace(d, 1 - d, 9)
        t_list = np.linspace(0, 1, 21)
        x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
        analytic = [self.u(t, x_mesh, y_mesh) for t in t_list]

        # calc and plot error
        errs = solver.calc_error(analytic)
        plt.plot(t_list, errs) # error grows with time if heat source is present
        plt.show()

    def test_error_diffusion(self):
        ic = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        u = lambda t, x, y: np.exp(-2*np.pi**2 * t) * ic(x, y)
        d = .1

        # numerical solution
        solver = Diffusion2D(domain=((0, 1), (0, 1)), d=0.1, ic=ic, bc='dirichlet', src=lambda t, x, y: 0)
        solver.solve(t_span=(0, 1), h=0.05, method='trapezoid')

        # analytical solution
        x, y = np.linspace(d, 1 - d, 9), np.linspace(d, 1 - d, 9)
        t_list = np.linspace(d, 1 - d, 21)
        x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
        analytic = [u(t, x_mesh, y_mesh) for t in t_list]

        # calc and plot error 
        errs = solver.calc_error(analytic)
        plt.plot(t_list, errs) # error decays with time if no heat source is present
        plt.show()


if __name__ == '__main__':
    unittest.main()
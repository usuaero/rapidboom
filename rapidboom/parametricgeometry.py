"""Tools for creating parametric surface descriptions and meshes."""
import numpy as np
from math import factorial
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gen_surface_grid(n1=50, n2=50, spacing='uniform'):
    """Generates a grid in the parametric space.

    The default is a 2D parametric space where each parameter ranges from
    0 to 1.

    Parameters
    ----------
    n1 : int
        The number of points for the first parameter.
    n2 : int
        The number of points for the second parameter.
    spacing : str
        The type of spacing used in distributing the points.

    Returns
    -------
    Parametric Grid : 3D numpy array
    """
    if spacing == 'uniform':
        p1 = np.linspace(0., 1., n1)
        p2 = np.linspace(0., 1., n2)
    elif spacing == 'cosine':
        p1 = cosine_spacing(0., 1., n1)
        p2 = cosine_spacing(0., 1., n2)
    else:
        raise RuntimeWarning(spacing+"spacing type not recognized")

    grid = np.zeros((n1, n2, 2))
    grid[:, :, 0] = p1[:, None]
    grid[:, :, 1] = p2[None, :]

    return grid


def cosine_spacing(start, stop, num_points, offset=0):
    # calculates the cosine spacing
    index = np.arange(num_points)
    points = .5*(1.-np.cos((np.pi/index[-1])*(index-offset)))
    points = start+(stop-start)*points

    return points


class Kulfan_Axisymmetric():
    """   """
    def __init__(self, D1, D2, shape_function):
        self._cross_section_cst = Kulfan_1D(N1=0.5, N2=0.5, shape_function=0.5)
        self._distribution_cst = Kulfan_1D(D1, D2, shape_function)
        self._length = 1.

    def set_length(self, length):
        self._length = length

    def evaluate(self, grid):
        n_psi, n_eta, d = grid.shape
        psi = grid[:, :, 0]
        eta = grid[:, :, 1]

        cart_coords = np.zeros((n_psi, n_eta, 3))

        L = self._length
        Sc = self._cross_section_cst.shape_function(eta)
        Cc = self._cross_section_cst.class_function(eta)
        Sd = self._distribution_cst.shape_function(psi)
        Cd = self._distribution_cst.class_function(psi)

        cart_coords[:, :, 0] = psi*L
        cart_coords[:, :, 1] = -Sd*Cd*(1.-2.*eta)*L
        cart_coords[:, :, 2] = Sd*Cd*Sc*Cc*L

        return cart_coords


class Kulfan_1D():
    """1D parametrization that can be used for airfoils or axisymmetric bodies.

    """
    def __init__(self, N1, N2, shape_function=None):
        self._N1 = N1
        self._N2 = N2
        if shape_function is not None:
            self._shape_function = shape_function
        else:
            self._shape_function = BernstienPolynomial(shape_param)

    def shape_function(self, parameter):
        # S = self._shape_function(parameter)
        try:
            S = self._shape_function(parameter)
        except TypeError:
            S = self._shape_function

        return S

    def class_function(self, parameter):

        return np.power(parameter, self._N1)*np.power(1.-parameter, self._N2)


class BernstienPolynomial():
    """ """
    def __init__(self, n_coeff, coefficients=None):
        self._n_coeff = n_coeff
        if coefficients is not None:
            self._coefficients = coefficients
        else:
            self._coefficients = n_coeff*[0.]
        self._K = self._calculate_k(n_coeff)

    @staticmethod
    def _calculate_k(number):
        K = number*[0]
        n = number-1
        for r in range(n+1):
            K[r] = factorial(n)/(factorial(r)*factorial(n-r))

        return K

    def set_coefficients(self, coeff):
        self._coefficients = coeff

    def __call__(self, parameter):

        A = self._coefficients
        K = self._K
        n = len(self._coefficients)-1
        F = np.zeros(parameter.shape)

        for r in range(n+1):
            F += A[r]*K[r]*np.power(parameter, r)*np.power(1.-parameter, n-r)

        # F = (A0*np.power(1.-parameter, 5.) +
        #      A1*5.*parameter*np.power(1.-parameter, 4.) +
        #      A2*10.*np.power(parameter, 2)*np.power(1.-parameter, 3.) +
        #      A3*10.*np.power(parameter, 3)*np.power(1.-parameter, 2.) +
        #      A4*5.*np.power(parameter, 4)*(1.-parameter) +
        #      A5*np.power(parameter, 5))

        return F


def plot_cart_surf(grid):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in np.arange(len(grid)):
        x = grid[i, :, 0]
        y = grid[i, :, 1]
        z = grid[i, :, 2]
        ax.scatter(x, y, z)

    plt.show()


class GaussianBump:
    """  """
    def __init__(self, height=1., location=0., standard_deviation=1.):
        self._height = height
        self._loc = location
        self._std = standard_deviation

    def set_parameters(self, peak_height, peak_loc, standard_deviation):
        self._height = peak_height
        self._loc = peak_loc
        self._std = standard_deviation

    def __call__(self, parameter):
        a = self._height
        b = self._loc
        c = self._std

        return a*np.exp(-0.5*np.power((parameter-b)/c, 2))


def constrain_ends(x_points):
    """Used to smoothly constrain the ends of a parameterization to zero.
    """
    x_front = x_points[0]
    x_back = x_points[-1]

    r = 0.1
    f_front = (2./np.pi)*np.arctan(r*x_points*x_points)
    f_back = (2./np.pi)*np.arctan(r*np.power((x_back-x_points), 2))
    f_constraint = f_front*f_back

    return f_constraint

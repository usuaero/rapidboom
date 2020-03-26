"""Implements equivalent area case."""
from rapidboom.sboomwrapper import SboomWrapper
import pyldb
import rapidboom.parametricgeometry as pg

import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

"""Supersonic aircraft equivalent area module

Routine Listings
-----------------
EquivArea(case_dir, sboom_exec, weather, altitude, deformation,
          area_filename, mach, phi, atmosphere_input):
          Class for calculating the sonic boom loudness of an aircraft
          using its equivalent area distribution and other flight
          parameters.
AreaConv(dp_filename, mach , ref_length, gamma, r_over_l):
         Class for converting a near-field pressure signature (dp/p) to
         and equivalent area distribution.

Notes
------
A number of different equivalent area distributions from different
flight conditions are available for use. A naming convention has been
established for varying Mach number, AoA, and azimuth angles. The list
included here provides the file names for the current Eq. Area dist
included with the github repository.

Baseline:
'mach1p600_aoa0p000_phi00p00.eqarea'

Varying Mach:
'mach1p550_aoa0p000_phi00p00.eqarea'
'mach1p650_aoaop000_phi00p00.eqarea'

Varying AoA:
'mach1p600_aoa0p200_phi00p00.eqarea'
'mach1p600_aoa-0p400_phi00p00.eqarea'

Varying AoA and Mach:
'mach1p583_aoa-0p273_phi00p00.eqarea'
'mach1p671_aoa0p392_phi00p00.eqarea'

Varying PHI:
'mach1p600_aoa0p000_phi02p00.eqarea'

The EquivArea class defaults to the baseline flight condition Eq. Area
"""

class EquivArea:
    def __init__(self, case_dir='./', sboom_exec='sboom_linux',
                 weather='standard', altitude=50000, deformation='gaussian',
                 area_filename = 'mach1p600_aoa0p000_phi00p00.eqarea',
                 mach=1.6, phi=0, ref_length = 32.92, r_over_l = 5,
                 atmosphere_input=None, elevation = 0):

        self.CASE_DIR = case_dir
        SBOOM_EXEC = sboom_exec
        REF_LENGTH = ref_length
        self.MACH = mach
        self.PHI = phi
        self.gamma = 1.4
        self.altitude = altitude
        self.deformation = deformation
        self.atmosphere_input = atmosphere_input
        self.eqa_filename = area_filename
        self.elevation = elevation
        R_over_L = r_over_l

        # INITIALIZE MODELS/TOOLS OF THE CASE AND SET ANY CONSTANT PARAMETERS
        # import equivalent area from file
        data_dir = os.path.join(os.path.dirname(__file__), "..", "misc")

        equiv_area_dist = np.genfromtxt(os.path.join(data_dir,
                                                     self.eqa_filename))

        self.position = equiv_area_dist[:, 0]
        self.area = equiv_area_dist[:, 1]

        # initialize sBOOM
        self._sboom = SboomWrapper(self.CASE_DIR, exe=SBOOM_EXEC)

        self._sboom.set(mach_number=self.MACH,
                        altitude=self.altitude,
                        propagation_start=R_over_L*REF_LENGTH*3.28084,
                        altitude_stop=self.elevation,
                        output_format=0,
                        input_xdim=2,
                        azimuthal_angles = self.PHI,
                        propagation_points=40000,
                        padding_points=8000)

        if weather != 'standard':
            # wind input (altitude ft, wind X, wind Y)
            if 'wind_x' in weather.keys():
                wind = []
                wind = weather['wind_x']  # data[key]['wind_y']]
                for i in range(len(wind)):
                    wind[i].append(weather['wind_y'][i][1])

                self._sboom.set(input_temp=weather['temperature'],
                                input_wind=wind,
                                input_humidity=weather['humidity'])
            else:
                self._sboom.set(input_temp=weather['temperature'],
                                input_humidity=weather['humidity'])
    def run(self, optimization_vars):
        def _runGaussian(area_temp, inputs):
            # unpack optimization variables
            gauss_amp, gauss_loc, gauss_std = inputs
            # evaluate change in Aeq at each location
            delta = pg.GaussianBump(gauss_amp, gauss_loc, gauss_std)
            delta_A = delta(self.position)
            # prevents nan issues when all variables are 0
            delta_A[np.isnan(delta_A)] = 0
            return delta_A*f_constraint

        def _runCubic(area_temp, inputs):
            # unpack optimization variables
            x, y, m, w0, w1 = inputs
            # evaluate change in Aeq at each location
            delta = pg.SplineBump(x, y, m, w0, w1)
            delta_A = delta(self.position)
            # prevents nan issues when all variables are 0
            delta_A[np.isnan(delta_A)] = 0
            return delta_A*f_constraint

        area_temp = self.area
        f_constraint = pg.constrain_ends(self.position)
    # generates gaussian function for each list of variables provided
    # to implement change in Aeq
        try:
            for var_list in optimization_vars:
                if self.deformation == 'gaussian':
                    area_temp = area_temp + _runGaussian(area_temp, var_list)
                elif self.deformation == 'cubic':
                    area_temp = area_temp + _runCubic(area_temp, var_list)
        # if one bump is provided
        except(TypeError):
            if self.deformation == 'gaussian':
                area_temp = area_temp + _runGaussian(area_temp, optimization_vars)
            elif self.deformation == 'cubic':
                area_temp = area_temp + _runCubic(area_temp, optimization_vars)

        self.new_equiv_area = np.array([self.position, area_temp]).T
        # np.savetxt('25D_equiv_area_dist_V1.txt', self.new_equiv_area, fmt='%.12f')

        # area units in ft^2 for sBoom input
        self.new_equiv_area[:, 1] = self.new_equiv_area[:, 1] * 10.7639
        # # plot new equivalent area
        # plt.plot(self.position, self.area)
        # plt.plot(self.new_equiv_area[:,0], (self.new_equiv_area[:,1])/(10.7639))
        # plt.title('Gaussian change in $A_E$$_q$, Amplitude: 0.03 $m^2$, Standard deviation: 0.5 m, Location: 30.0 m', fontsize = 16)
        # plt.xlabel("Axial position(m)", fontsize = 16)
        # plt.ylabel('$A_E$$_q$ ($m^2$)', fontsize = 16)
        # plt.legend(['Baseline $A_E$$_q$', 'Modified $A_E$$_q$'], fontsize = 16)
        # plt.xlim((0, 50))
        # plt.show()

        # update sBOOM settings and run
        self._sboom.set(signature=self.new_equiv_area, input_format=2)
        sboom_results = self._sboom.run(atmosphere_input=self.atmosphere_input)
        g_sig = sboom_results["signal_0"]["ground_sig"]
        # # self.ground_sig = g_sig
        # plt.plot(g_sig[:, 0], g_sig[:, 1])
        # plt.show()

        # grab the loudness level
        # noise_level = sboom_results["signal_0"]["C_weighted"]
        noise_level = pyldb.perceivedloudness(g_sig[:, 0],
                                              g_sig[:, 1],
                                              pad_rear=4)

        return noise_level

class AreaConv:
    '''Class for converting a near-field dp/p to an equivalent area distribution.

    This code is based on Eq.(11) found in the reference by Wu Li and Sriram
    Rallabhandi included below. This tool can be used to conver a
    supersonic aircrafts nearfield pressure signature, dp/p(x), to an equivalent
    area distribution representation of the aircraft. The user simply needs to
    create an instance of the area conversion class and use the class run method.

    Methods
    --------

    run(self, save_filename, write = True):
    Calls all class functions to produce and write the equivalent area to file.

    Notes
    ------
    The input file should contain two arrays of data, the first being sensor
    locations and the second being the dp/p(x) values for each location. The
    user only needs to specify the filename of the dp/p distribution when
    creating an instance of the class. Defaults are set for the aircraft
    flight parameters but can easily be changed by passing the desired parameters
    at the creation of the class.

    To generate the equivalent area distribution and save the data to file,
     the user needs to call the class run method with a desired output filename.
    The user should be aware of the units that are used in the dp/p(x)
    distribution. The area returned will be the square of the x position units.
    Please ignore the runtime warning for an invalid value encountered in
    taking the sqrt, this is expected and addressed within the code.

    See Also
    --------
    numpy.isnan : Returns indices of array holding nan
    scipy.integrate.trapz : Performs trapezoidal numerical integration
    numpy.newaxis: and the use of None in array slicing

    References
    -----------
    Li, W., and Rallabhandi, S. K., “Inverse Design of Low-Boom Supersonic Concepts
    Using Reversed Equivalent-Area Targets,”Journal of Aircraft, Vol. 51, No. 1,
    2014, pp. 29–36.

    Example
    --------
    from rapidboom import AreaConv

    class_name = AreaConv(dp_filename = 'Euler_UndertrackNF_RL5', mach = 1.6,
                          ref_length = 32.92, gamma = 1.4, r_over_l = 5)
    eq_area_dist = class_name.run(save_filename = 'test.eqarea', write = True)

    '''

    def __init__(self, dp_filename, mach = 1.6, ref_length = 32.92,
                 gamma = 1.4, r_over_l = 5):
        '''Initializes equivalent area class instance.

        Parameters
        -----------
        dp_filename: string
            Name of dp/p distribution file.
        mach: float, optional
            Aircraft Mach number.
        ref_length: float, optional
            Aircraft reference length.
        gamma: float, optional
            Ratio of specifc heats.
        r_over_l: int, optional
            Near-field sensor location ratio.

        '''
        # pull nearfield data from file
        self._dp_p_dist = np.genfromtxt(dp_filename)
        self._dp_p = self._dp_p_dist[:,1]
        self._x = self._dp_p_dist[:,0]

        self.MACH = mach # cruise Mach number
        self.ref_len = ref_length # meters, reference length of aircraft
        self.GAMMA = gamma # ratio of specific heats
        self.R_over_L = r_over_l # sensor distance as ratio of ref_len

        # near field sensor distance
        self._near_field_dist = self.ref_len*self.R_over_L

    def _convert(self):
        # calculates coefficient of Eq.(11)
        coeff1_num = 4*np.sqrt((2*self._near_field_dist*np.sqrt((self.MACH**2)-1)))
        coeff1_denom = self.GAMMA*(self.MACH**2)
        coeff1 = coeff1_num/coeff1_denom

        # performs integration and calculates EqA at each dp/p location
        integrand = (self._dp_p[None,:])*(np.sqrt(self._x[:,None] - self._x[None,:]))
        print("Expected RuntimeWarning, please ignore.")
        integrand[np.isnan(integrand)] = 0 # replaces nan with 0
        area = coeff1*integrate.trapz(integrand[:, :], self._x)

        # organize and return data
        data = np.array([self._x, area]).T
        return data

    # function defined to run the area conversion class functions
    def run(self, save_filename, write = True):
        '''Calls class methods to convert EqA distribution and save the file.

        Parameters
        -----------
        save_filename: string
                Desired name for equivalent area distribution .txt file.
        write: boolean
                user input flag to write Eqa to file or only return data
        '''
        # generate area data
        area_data = self._convert()
        # write data to file with write flag is true
        if write:
            np.savetxt(save_filename, area_data, fmt='%.12f')

        return area_data

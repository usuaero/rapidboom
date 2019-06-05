"""Implements equivalent area case."""
from rapidboom.sboomwrapper import SboomWrapper
import pyldb
import rapidboom.parametricgeometry as pg
import os

import numpy as np
import matplotlib.pyplot as plt


class EquivArea:
    def __init__(self, case_dir='./', sboom_exec='sboom_linux',
                 weather='standard', altitude=50000, deformation='gaussian',
                 atmosphere_input=None):
        self.CASE_DIR = case_dir
        SBOOM_EXEC = sboom_exec
        REF_LENGTH = 32.92
        self.MACH = 1.6
        # self.MACH = 1.583
        # self.MACH = 1.671
        self.aoa = 0.
        self.gamma = 1.4
        self.altitude = altitude
        self.deformation = deformation
        self.atmosphere_input = atmosphere_input
        R_over_L = 5

        # INITIALIZE MODELS/TOOLS OF THE CASE AND SET ANY CONSTANT PARAMETERS
        # import equivalent area from file
        data_dir = os.path.join(os.path.dirname(__file__), "..", "misc")

        equiv_area_dist = np.genfromtxt(os.path.join(data_dir,
                                                     "sBoom__DPatR_3_clean_phi00.00.eqarea"))
        # "mach1p583_aoa-0p273.eqarea"))
        # "mach1p671_aoa0p392.eqarea"))

        self.position = equiv_area_dist[:, 0]
        self.area = equiv_area_dist[:, 1]

        # initialize sBOOM
        self._sboom = SboomWrapper(self.CASE_DIR, exe=SBOOM_EXEC)

        self._sboom.set(mach_number=self.MACH,
                        altitude=self.altitude,
                        propagation_start=R_over_L*REF_LENGTH*3.28084,
                        altitude_stop=0.,
                        output_format=0,
                        input_xdim=2,
                        propagation_points=40000,
                        padding_points=8000)

        if weather != 'standard':
            # wind input (altitude ft, wind X, wind Y)
            wind = []
            wind = weather['wind_x']  # data[key]['wind_y']]
            for i in range(len(wind)):
                wind[i].append(weather['wind_y'][i][1])

            self._sboom.set(input_temp=weather['temperature'],
                            input_wind=wind,
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

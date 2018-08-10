"""Implements axie bump case."""
from rapidboom.sboomwrapper import SboomWrapper
import pyldb
import rapidboom.parametricgeometry as pg
import panairwrapper
import os

import numpy as np
# import matplotlib.pyplot as plt


class AxieBump:
    def __init__(self, case_dir='./', panair_exec='panair',
                 sboom_exec='sboom_linux'):
        CASE_DIR = case_dir
        PANAIR_EXEC = panair_exec
        SBOOM_EXEC = sboom_exec
        REF_LENGTH = 32.92
        MACH = 1.6
        gamma = 1.4
        R_over_L = 1

        self.MESH_COARSEN_TOL = 0.000035
        self.N_TANGENTIAL = 20

        # INITIALIZE MODELS/TOOLS OF THE CASE AND SET ANY CONSTANT PARAMETERS
        # import AXIE geometry from file
        data_dir = os.path.join(os.path.dirname(__file__), "..", "misc")
        geometry = np.genfromtxt(os.path.join(data_dir, "axie-geom-v1-mm.dat"))
        self._x_geom = geometry[:, 0]
        self._r_geom = geometry[:, 1]
        self._x_geom *= 0.001  # meters
        self._r_geom *= 0.001  # meters

        # initialize Panair
        self._panair = panairwrapper.PanairWrapper('axie', CASE_DIR,
                                                   exe=PANAIR_EXEC)
        self._panair.set_aero_state(MACH)
        self._panair.set_sensor(MACH, R_over_L, REF_LENGTH)
        self._panair.set_symmetry(1, 1)
        self._panair.add_network('axie_surface', None)
        # panair.add_sensor(r_over_l=3)

        # initialize sBOOM
        self._sboom = SboomWrapper(CASE_DIR, exe=SBOOM_EXEC)
        self._sboom.set(mach_number=MACH,
                        altitude=51706.037,
                        propagation_start=R_over_L*REF_LENGTH*3.28084,
                        altitude_stop=0.,
                        output_format=0,
                        input_xdim=2)

    def run(self, optimization_var):
        # unpack optimization variables and assign to appropriate locations
        bump_height, bump_loc, bump_width = optimization_var

        # evaluate bump at x coordinates and add to radius values
        bump = pg.GaussianBump(bump_height, bump_loc, bump_width)
        f_constraint = pg.constrain_ends(self._x_geom)
        # plt.plot(x_geom, f_constraint)
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.show()
        r_bump = bump(self._x_geom)*f_constraint
        r_total = self._r_geom+r_bump

        # plot geometry
        # plt.plot(x_geom, r_geom)
        # plt.plot(x_geom, r_total)
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.show()

        # coarsen grid based on curvature
        x_final, r_final = panairwrapper.mesh_tools.coarsen_axi(self._x_geom, r_total,
                                                                self.MESH_COARSEN_TOL, 5.)

        # pass in the new R(x) into panair axie surface function
        networks = panairwrapper.mesh_tools.axisymmetric_surf(x_final, r_final, self.N_TANGENTIAL)

        # update Panair settings and run
        self._panair.clear_networks()
        for i, n in enumerate(networks):
            self._panair.add_network('surface'+str(i), n)
        try:
            panair_results = self._panair.run()
        except RuntimeError:
            # if panair blows up, return default high value
            return 999.

        offbody_data = panair_results.get_offbody_data()
        distance_along_sensor = offbody_data[:, 2]
        dp_over_p = 0.5*gamma*MACH**2*offbody_data[:, -2]
        nearfield_sig = np.array([distance_along_sensor, dp_over_p]).T
        # plt.plot(nearfield_sig[:, 0], nearfield_sig[:, 1])
        # plt.show()

        # update sBOOM settings and run
        self._sboom.set(signature=nearfield_sig)
        sboom_results = self._sboom.run()
        ground_sig = sboom_results["signal_0"]["ground_sig"]

        # grab the loudness level
        # noise_level = sboom_results["signal_0"]["C_weighted"]
        noise_level = pyldb.perceivedloudness(ground_sig[:, 0], ground_sig[:, 1])

        return noise_level

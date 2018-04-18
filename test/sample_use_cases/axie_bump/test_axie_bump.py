"""Tests the sample cases."""

import pytest

from rapidboom.sboomwrapper import SboomWrapper
import rapidboom.parametricgeometry as pg
import panairwrapper

import numpy as np
import matplotlib.pyplot as plt


# folder where all case files for the tools will be stored
CASE_DIR = "./test/sample_use_cases/axie_bump/"

REF_LENGTH = 32.92
MACH = 1.6
R_over_L = 1

# INITIALIZE MODELS/TOOLS OF THE CASE AND SET ANY CONSTANT PARAMETERS
# import AXIE geometry from file
geometry = np.genfromtxt(CASE_DIR+"axie-geom-v1-mm.dat")
x_geom = geometry[:, 0]
r_geom = geometry[:, 1]
x_geom *= 0.001  # meters
r_geom *= 0.001  # meters

# initialize Panair
panair = panairwrapper.PanairWrapper('axie', CASE_DIR)
panair.set_aero_state(MACH)
panair.set_sensor(MACH, R_over_L, REF_LENGTH)
panair.set_symmetry(1, 1)
panair.add_network('axie_surface', None)
# panair.add_sensor(r_over_l=3)

# initialize sBOOM
sboom = SboomWrapper(CASE_DIR)
sboom.set(mach_number=MACH,
          altitude=51706.037,
          propagation_start=R_over_L*REF_LENGTH*3.28084,
          altitude_stop=0.,
          output_format=0,
          input_xdim=2)


def run(optimization_var):
    # unpack optimization variables and assign to appropriate locations
    bump_height, bump_loc, bump_width = optimization_var

    # evaluate bump at x coordinates and add to radius values
    bump = pg.GaussianBump(bump_height, bump_loc, bump_width)
    f_constraint = pg.constrain_ends(x_geom)
    # plt.plot(x_geom, f_constraint)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.show()
    r_bump = bump(x_geom)*f_constraint
    r_total = r_geom+r_bump

    # plot geometry
    # plt.plot(x_geom, r_geom)
    # plt.plot(x_geom, r_total)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.show()

    # coarsen grid based on curvature
    x_final, r_final = panairwrapper.mesh_tools.coarsen_axi(x_geom, r_total,
                                                            0.00005, 5.)

    # pass in the new R(x) into panair axie surface function
    networks = panairwrapper.mesh_tools.axisymmetric_surf(x_final, r_final, 10)

    # update Panair settings and run
    panair.clear_networks()
    for i, n in enumerate(networks):
        panair.add_network('surface'+str(i), n)
    try:
        panair_results = panair.run()
    except RuntimeError:
        # if panair blows up, return default high value
        return 120

    offbody_data = panair_results.get_offbody_data()
    nearfield_sig = np.array([offbody_data[:, 2], 1.792*offbody_data[:, -2]]).T
    # plt.plot(nearfield_sig[:, 0], nearfield_sig[:, 1])
    # plt.show()

    # update sBOOM settings and run
    sboom.set(signature=nearfield_sig)
    sboom_results = sboom.run()

    # grab the loudness level
    noise_level = sboom_results["signal_0"]["C_weighted"]

    return noise_level


def test_axie_bump_case():
    loudness = run([0.1, 20., 6.])

    print("C-weighted loudness", loudness)
    assert(loudness - 93.54010125368868 < 0.01)

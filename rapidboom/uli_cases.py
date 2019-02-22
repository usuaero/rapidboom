"""Implements axie bump case."""
from rapidboom.sboomwrapper import SboomWrapper
import pyldb
import rapidboom.parametricgeometry as pg
import panairwrapper
import panairwrapper.mesh_tools as meshtools
from aeropy.CST_3D.module import *
import aeropy.CST_3D as cst
from aeropy.filehandling.vtk import generate_surface
import os

import numpy as np
import matplotlib.pyplot as plt


class AxieBump:
    def __init__(self, case_dir='./', panair_exec='panair',
                 sboom_exec='sboom_linux', weather='standard',
                 altitude=45000):
        CASE_DIR = case_dir
        PANAIR_EXEC = panair_exec
        SBOOM_EXEC = sboom_exec
        REF_LENGTH = 32.92
        self.MACH = 1.6
        self.aoa = 0.
        self.gamma = 1.4
        self.altitude = altitude
        R_over_L = 5

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
        self._panair = panairwrapper.PanairWrapper('wingbody', CASE_DIR,
                                                   exe=PANAIR_EXEC)
        self._panair.set_aero_state(self.MACH, self.aoa)
        self._panair.set_sensor(self.MACH, self.aoa, R_over_L, REF_LENGTH)
        self._panair.set_symmetry(1, 1)
        # self._panair.add_network('axie_surface', None)
        # panair.add_sensor(r_over_l=3)

        # initialize sBOOM
        self._sboom = SboomWrapper(CASE_DIR, exe=SBOOM_EXEC)

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

        f_constraint = pg.constrain_ends(self._x_geom)
        r_total = self._r_geom

        # generates bump for each list of variables provided
        for var_list in optimization_vars:
            # unpack optimization variables and assign to appropriate locations
            bump_height, bump_loc, bump_width = var_list

            # evaluate bump at x coordinates and add to radius values
            bump = pg.GaussianBump(bump_height, bump_loc, bump_width)
            #bump = pg.WedgeBump(bump_height, bump_loc, bump_width)
            # plt.plot(x_geom, f_constraint)
            # plt.gca().set_aspect('equal', 'datalim')
            # plt.show()
            r_total = r_total + bump(self._x_geom)*f_constraint

        # plot geometry
        # plt.plot(self._x_geom, self._r_geom)
        # plt.plot(self._x_geom, r_total)
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.show()

        # coarsen grid based on curvature
        x_final, r_final = panairwrapper.mesh_tools.coarsen_axi(self._x_geom, r_total,
                                                                self.MESH_COARSEN_TOL, 5.)

        # pass in the new R(x) into panair axie surface function
        self._networks = panairwrapper.mesh_tools.axisymmetric_surf(
            x_final, r_final, self.N_TANGENTIAL)

        # update Panair settings and run
        self._panair.clear_networks()
        for i, n in enumerate(self._networks):
            self._panair.add_network('surface'+str(i), n, xy_indexing=True)
        try:
            panair_results = self._panair.run(overwrite=True)
        except RuntimeError:
            # if panair blows up, return default high value
            return 999.

        offbody_data = panair_results.get_offbody_data()
        distance_along_sensor = offbody_data[:, 2]
        dp_over_p = 0.5*self.gamma*self.MACH**2*offbody_data[:, -2]
        nf_sig = np.array([distance_along_sensor, dp_over_p]).T
        # plt.plot(nearfield_sig[:, 0], nearfield_sig[:, 1])
        # plt.show()
        self.nearfield_sig = nf_sig
        # update sBOOM settings and run
        self._sboom.set(signature=nf_sig)
        sboom_results = self._sboom.run()
        g_sig = sboom_results["signal_0"]["ground_sig"]
        # plt.plot(g_sig[:, 0], g_sig[:, 1])
        # plt.show()
        self.ground_sig = g_sig

        # grab the loudness level
        # noise_level = sboom_results["signal_0"]["C_weighted"]
        noise_level = pyldb.perceivedloudness(g_sig[:, 0], g_sig[:, 1], pad_rear=4)

        return noise_level

    def generate_vtk(self):
        for i, n in enumerate(self._networks):
            generate_surface(n, "surface"+str(i))


class DeltaWing:
    def __init__(self, case_dir='./', panair_exec='panair',
                 sboom_exec='sboom_linux'):

        # initialize geometry
        Au = np.array([0.172802, 0.167353, 0.130747, 0.172053, 0.112797, 0.168891])
        Al = np.array([0.163339, 0.175407, 0.134176, 0.152834, 0.133240, 0.161677])

        span = 2.
        chord_r = 1.
        chord_t = .3
        self._cp = ControlPoints()
        self._cp.set(half_span=span/2.,
                     chord=[chord_r, chord_t],
                     twist=[0, 0],
                     shear=[0, .1],
                     sweep=[0, 2.],
                     N1=[1., 1.],
                     N2=[1., 1.])

        self._B = {'upper': [Au, Au], 'lower': [Al, Al]}

        # generate parameter mesh
        self._N_psi = 20
        self._N_eta = 40
        upper_psi, upper_eta = meshtools.meshparameterspace(
            (self._N_psi, self._N_eta), flip=True, cos_spacing=True)
        lower_psi, lower_eta = meshtools.meshparameterspace(
            (self._N_psi, self._N_eta), cos_spacing=True)
        self._mesh = {'upper': np.concatenate([upper_psi.reshape(1, np.size(upper_psi)),
                                               upper_eta.reshape(1, np.size(upper_eta))]).T,
                      'lower': np.concatenate([lower_psi.reshape(1, np.size(lower_psi)),
                                               lower_eta.reshape(1, np.size(lower_eta))]).T}

        # initialize Panair
        self.gamma = 1.4
        self.MACH = 1.6
        self.aoa = 0.
        R_over_L = 5.
        REF_LENGTH = span
        self._panair = panairwrapper.PanairWrapper('deltawing', case_dir,
                                                   exe=panair_exec)
        self._panair.set_aero_state(self.MACH)
        self._panair.set_sensor(self.MACH, R_over_L, REF_LENGTH, 4.)
        self._panair.set_symmetry(1, 0)
        chord_avg = (chord_r+chord_t)/2.
        planform_area = span*chord_avg
        self._panair.set_reference_data(planform_area, span, chord_avg)

        # initialize sBOOM
        self._sboom = SboomWrapper(case_dir, exe=sboom_exec)
        self._sboom.set(mach_number=self.MACH,
                        altitude=51706.037,
                        propagation_start=R_over_L*REF_LENGTH*3.28084,
                        altitude_stop=0.,
                        output_format=0,
                        input_xdim=2,
                        propagation_points=25000,
                        padding_points=12000)

    def run(self, optimization_var):
        # unpack optimization variables and assign to appropriate locations
        bump_height, bump_loc, bump_width = optimization_var

        # generate panair mesh
        output = CST_3D(self._B, self._mesh, cp=self._cp, mesh_type='parameterized')
        upper_xyz = output["upper"]
        lower_xyz = output["lower"]
        upper_xyz.resize((self._N_psi, self._N_eta, 3))
        lower_xyz.resize((self._N_psi, self._N_eta, 3))

        cap_mesh = np.zeros((self._N_psi, 2, 3))
        cap_mesh[:, 0, :] = lower_xyz[:, -1, :]
        cap_mesh[:, 1, :] = np.flip(upper_xyz[:, -1, :], 0)

        trailing_edge = upper_xyz[0, :, :]
        wake = meshtools.generate_wake(trailing_edge, 10.,
                                       angle_of_attack=self.aoa)

        # update Panair settings and run
        self._panair.clear_networks()
        self._panair.add_network("upper", upper_xyz)
        self._panair.add_network("lower", lower_xyz)
        self._panair.add_network("cap", cap_mesh)
        self._panair.add_network("wake", wake, 18.)

        try:
            panair_results = self._panair.run()
            panair_results.write_vtk()
            forces_moments = panair_results.get_forces_and_moments()
            CL = forces_moments["cl"]
            CD = forces_moments["cdi"]
        except RuntimeError:
            # if panair blows up, return default high value
            return 999.

        offbody_data = panair_results.get_offbody_data()
        distance_along_sensor = offbody_data[:, 2]
        dp_over_p = 0.5*self.gamma*self.MACH**2*offbody_data[:, -2]
        nearfield_sig = np.array([distance_along_sensor, dp_over_p]).T
        plt.plot(nearfield_sig[:, 0], nearfield_sig[:, 1])
        plt.title("nearfield signature")
        plt.show()

        # update sBOOM settings and run
        self._sboom.set(signature=nearfield_sig)
        sboom_results = self._sboom.run()
        ground_sig = sboom_results["signal_0"]["ground_sig"]
        plt.plot(ground_sig[:, 0], ground_sig[:, 1])
        plt.title("ground signature")
        plt.show()

        # grab the loudness level
        # noise_level = sboom_results["signal_0"]["C_weighted"]
        noise_level = pyldb.perceivedloudness(ground_sig[:, 0], ground_sig[:, 1])

        return noise_level, CL, CD


class WingBody:
    def __init__(self, case_dir='./', panair_exec='panair',
                 sboom_exec='sboom_linux'):
        # initialize Panair
        self.gamma = 1.4
        self.MACH = 1.6
        self.aoa = 0.
        self.R_over_L = 5.
        self._panair = panairwrapper.PanairWrapper('deltawing', case_dir,
                                                   exe=panair_exec)
        self._panair.set_aero_state(self.MACH)
        self._panair.set_symmetry(1, 0)

        # initialize sBOOM
        self._sboom = SboomWrapper(case_dir, exe=sboom_exec)
        self._sboom.set(mach_number=self.MACH,
                        altitude=51706.037,
                        altitude_stop=0.,
                        output_format=0,
                        input_xdim=2,
                        propagation_points=15000,
                        padding_points=5000)

    def run(self, **variables):
        span = variables.get('span', 2.)
        eta_cp = [0., 1.]
        chord = variables.get('chord', [1.5, 0.4])
        sweep_angle = variables.get('sweep', 56.)
        sweep = [0., -np.tan(sweep_angle*np.pi/180.)]
        dihedral_angle = variables.get('dihedral', 0.)
        dihedral_u = [0., np.tan(dihedral_angle*np.pi/180.)]
        dihedral_l = [0., -np.tan(dihedral_angle*np.pi/180.)]

        # fuselage parameters
        length = 4.

        f_sx = cst.piecewise_linear(eta_cp, chord)
        f_sy_upper = cst.BernstienPolynomial(5, [0.172802, 0.167353, 0.130747,
                                                 0.172053, 0.112797, 0.168891])
        f_sy_lower = cst.BernstienPolynomial(5, [0.163339, 0.175407, 0.134176,
                                                 0.152834, 0.133240, 0.161677])
        f_etashear = cst.piecewise_linear(eta_cp, sweep)
        f_zetashear_u = cst.piecewise_linear(eta_cp, dihedral_u)
        f_zetashear_l = cst.piecewise_linear(eta_cp, dihedral_l)

        wing_upper = cst.CST3D(rotation=(0., 0., 90.),
                               location=(1.5, 0., 0.),
                               XYZ=(span/2., 1., 1.),
                               ref=(0., 1., 0.),
                               sx=f_sx,
                               nx=(0., 0.),
                               sy=f_sy_upper,
                               ny=(1., 1.),
                               etashear=f_etashear,
                               zetashear=f_zetashear_u)

        wing_lower = cst.CST3D(rotation=(0., 0., 90.),
                               location=(1.5, 0., 0.),
                               XYZ=(span/2., 1., -1.),
                               ref=(0., 1., 0.),
                               sx=f_sx,
                               nx=(0., 0.),
                               sy=f_sy_lower,
                               ny=(1., 1.),
                               etashear=f_etashear,
                               zetashear=f_zetashear_l)

        fuselage = cst.CST3D(rotation=(-90., 0., 0.),
                             XYZ=(length, 1.5, 1.5),
                             nx=(1., 1.),
                             ref=(0., 0.5, 0.),
                             ny=(.5, .5))

        # generate mesh
        N_chord = 5
        N_span = 10

        eta_spacing_w = meshtools.cosine_spacing(0., 1., N_chord)

        # upper wing intersection and mesh
        p_intersect_u = cst.intersection(wing_upper, fuselage, eta_spacing_w, 0.3)
        np.savetxt("intersection_upper.csv", p_intersect_u)

        psi_wu_intrsct, eta_wu_intrsct = wing_upper.inverse(p_intersect_u[:, 0],
                                                            p_intersect_u[:, 1],
                                                            p_intersect_u[:, 2])
        psi_limit_wu = np.array([psi_wu_intrsct, eta_spacing_w]).T
        psi_wu, eta_wu = meshtools.meshparameterspace((N_span, N_chord), flip=False, cos_spacing=True,
                                                      psi_limits=(psi_limit_wu, None))

        mesh_wu = wing_upper(psi_wu, eta_wu)

        # lower wing intersection and mesh
        p_intersect_l = cst.intersection(wing_lower, fuselage, eta_spacing_w, 0.3)
        np.savetxt("intersection_lower.csv", p_intersect_l)

        psi_wl_intrsct, eta_wl_intrsct = wing_lower.inverse(p_intersect_l[:, 0],
                                                            p_intersect_l[:, 1],
                                                            p_intersect_l[:, 2])
        psi_limit_wl = np.array([psi_wl_intrsct, eta_spacing_w]).T
        psi_wl, eta_wl = meshtools.meshparameterspace((N_span, N_chord), flip=False, cos_spacing=True,
                                                      psi_limits=(psi_limit_wl, None))

        mesh_wl = wing_lower(psi_wl, eta_wl)

        # fuselage intersections and mesh
        N_nose = 3
        N_tail = 3
        N_circ = 5
        psi_fu_intersect, eta_fu_intersect = fuselage.inverse(p_intersect_u[:, 0],
                                                              p_intersect_u[:, 1],
                                                              p_intersect_u[:, 2])
        psi_fl_intersect, eta_fl_intersect = fuselage.inverse(p_intersect_l[:, 0],
                                                              p_intersect_l[:, 1],
                                                              p_intersect_l[:, 2])
        psi_frontpoint = psi_fu_intersect[-1]
        eta_frontpoint = eta_fu_intersect[-1]
        psi_rearpoint = psi_fu_intersect[0]
        eta_rearpoint = eta_fu_intersect[0]

        front_section_fuse = np.full((N_nose, 2,), 0.)
        rear_section_fuse = np.full((N_tail, 2,), 0.)

        front_section_fuse[:, 0] = np.flipud(np.linspace(0., psi_frontpoint, N_nose))
        front_section_fuse[:, 1] = eta_frontpoint
        rear_section_fuse[:, 1] = eta_rearpoint
        # rear_section_fuse[:, 1] = np.linspace(rear_point_fuse[1], 1., N_tail)
        rear_section_fuse[:, 0] = np.flipud(meshtools.cosine_spacing(psi_rearpoint, 1., N_tail))

        int_upperfuse_p = np.concatenate(
            (rear_section_fuse[:-1], np.array([psi_fu_intersect, eta_fu_intersect]).T, front_section_fuse[1:]))
        int_lowerfuse_p = np.concatenate(
            (rear_section_fuse[:-1], np.array([psi_fl_intersect, eta_fl_intersect]).T, front_section_fuse[1:]))

        psi_fu, eta_fu = meshtools.meshparameterspace((N_chord+N_nose+N_tail-2, N_circ), flip=True,
                                                      eta_limits=(None, np.flipud(int_upperfuse_p)),
                                                      cos_spacing=False)
        psi_fl, eta_fl = meshtools.meshparameterspace((N_chord+N_nose+N_tail-2, N_circ), flip=True,
                                                      eta_limits=(np.flipud(int_lowerfuse_p), None),
                                                      cos_spacing=False)

        for j in range(N_circ):
            psi_fu[:, j] = int_upperfuse_p[:, 0]
            psi_fl[:, j] = int_lowerfuse_p[:, 0]

        mesh_fu = fuselage(psi_fu, eta_fu)
        mesh_fl = fuselage(psi_fl, eta_fl)

        network_wu = np.dstack(mesh_wu)
        network_wl = np.dstack(mesh_wl)
        network_fu = np.dstack(mesh_fu)
        network_fl = np.dstack(mesh_fl)

        # generate cap
        network_cap = np.zeros((N_chord, 2, 3))
        network_cap[:, 0, :] = network_wl[-1, :, :]
        network_cap[:, 1, :] = network_wu[-1, :, :]

        # calculate wake
        wing_trailing_edge = network_wu[:, 0, :]
        fuselage_wake_boundary = fuselage(rear_section_fuse[:, 0], rear_section_fuse[:, 1])

        fuselage_wake_boundary = np.flipud(np.array([fuselage_wake_boundary[0],
                                                     fuselage_wake_boundary[1],
                                                     fuselage_wake_boundary[2]]).T)

        inner_endpoint = np.copy(fuselage_wake_boundary[-1])
        n_wake_streamwise = len(fuselage_wake_boundary)
        wake = meshtools.generate_wake(wing_trailing_edge, inner_endpoint[0],
                                       n_wake_streamwise, self.aoa, cos_spacing=True)

        wingbody_wake = np.zeros((n_wake_streamwise, 2, 3))
        wingbody_wake[:, 0] = fuselage_wake_boundary
        wingbody_wake[:, 1] = wake[:, 0]

        # run in Panair
        chord_avg = (chord[0]+chord[1])/2.
        planform_area = span*chord_avg
        self._panair.set_reference_data(planform_area, span, chord_avg)
        self._panair.add_network("wing_u", np.flipud(network_wu))
        self._panair.add_network("wing_l", network_wl)
        self._panair.add_network("fuselage_u", network_fu)
        self._panair.add_network("fuselage_l", network_fl)
        self._panair.add_network("wing_cap", np.flipud(network_cap))
        self._panair.add_network("wake", wake, 18.)
        self._panair.add_network("wingbody_wake", wingbody_wake, 20.)

        self._panair.set_sensor(self.MACH, self.aoa, 5, 2.5*length, 1.)

        try:
            self.panair_results = self._panair.run()
            forces_moments = self.panair_results.get_forces_and_moments()
            CL = forces_moments["cl"]
            CD = forces_moments["cdi"]
        except RuntimeError:
            # if panair blows up, return default high value
            return 999., 999., 999.

        offbody_data = self.panair_results.get_offbody_data()
        distance_along_sensor = offbody_data[:, 2]
        dp_over_p = 0.5*self.gamma*self.MACH**2*offbody_data[:, -2]
        nearfield_sig = np.array([distance_along_sensor, dp_over_p]).T
        plt.plot(nearfield_sig[:, 0], nearfield_sig[:, 1])
        plt.title("nearfield signature")
        plt.show()
        np.savetxt('nearfield_sig', nearfield_sig)

        # update sBOOM settings and run
        self._sboom.set(propagation_start=self.R_over_L*span*3.28084)
        self._sboom.set(signature=nearfield_sig)
        sboom_results = self._sboom.run()
        ground_sig = sboom_results["signal_0"]["ground_sig"]
        plt.plot(ground_sig[:, 0], ground_sig[:, 1])
        plt.title("ground signature")
        plt.show()

        # grab the loudness level
        # noise_level = sboom_results["signal_0"]["C_weighted"]
        noise_level = pyldb.perceivedloudness(ground_sig[:, 0], ground_sig[:, 1])

        return noise_level, CL, CD

    def generate_vtk(self):
        self.panair_results.write_vtk()

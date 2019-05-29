"""This module provides a Python wrapper for the sBOOM program.

The primary purpose of this module is to provide a programmatic
interface) for sBOOM. This is done through the sBOOM interface available to
us, the input and output files. Thus, this module handles all of the formatting
details of generating an sBOOM inputfile and extracting data from the
outputfiles.

Additionaly, this module seeks to provide sane defaults for the many settings
available in sBOOM.

Example
-------
from rapidboom.sboomwrapper import SboomWrapper


# near-field signature data
sig_data = np.genfromtxt('./test/testfiles/inputtest.data', skip_header=3)

# generate case and set case parameters
sboom = SboomWrapper('./test_case')
sboom._sboom_exec = "sboom_linux"  # name of the sboom executable
sboom.set(signature=sig_data,
          mach_number=1.6,
          altitude=45000.,
          propagation_start=200.,
          altitude_stop=4500.)

# run case and retrieve results
results = basic_test_case.run()
a_weighted = results["signal_0"]["A_weighted"]
c_weighted = results["signal_0"]["C_weighted"]
ground_sig = results["signal_0"]["ground_sig"]

Notes
-----
Currently, the sBOOM executable needs to be copied into the directory where
python is being run. In the future, the location of the sBOOM executable will
need to be specified when installing sboomwrapper instead of needing to copy
it into the directory where python is being called from.

It should also be noted that this is a work in progress so not all of the
functionaltity of sBOOM has been built into this interface yet. Feel free to
request or contribute any desired additions!

The use of this module is obviously based on sBOOM already being
installed on your system. sBOOM must be requested from NASA at
http://www.software.nasa.gov

"""
from collections import OrderedDict
import numpy as np
import os
import sys
import ctypes
import subprocess
import shutil
import time


class SboomWrapper:
    """The primary access point for specifying and running a case.

    Parameters
    ----------
    directory : str
        All the files generated for the case will be stored in an sBOOM folder
        in this directory.

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, directory, exe="sboom_linux"):
        # self._title = title
        self._directory = directory+"/sBOOM/"
        self._parameters = self._init_parameters()
        self._results = {}
        self._sboom_exec = exe
        self._sboom_loc = os.path.join(os.path.dirname(__file__), "..")

    def _init_parameters(self):
        parameters = OrderedDict([["signature", None],
                                  ["signature_filename", "signature.data"],
                                  ["mach_number", None],
                                  ["altitude", None],
                                  ["propagation_start", None],
                                  ["propagation_points", 10000],
                                  ["padding_points", 1200],
                                  ["nonlinear", True],
                                  ["thermoviscous", True],
                                  ["relaxation", True],
                                  ["step_size", 0.01],
                                  ["reflection_factor", 1.9],
                                  ["resampled_points", 1000],
                                  ["padding_tolerance", 0.001],
                                  ["input_tolerance", 1e-6],
                                  ["input_format", 0],
                                  ["altitude_stop", None],
                                  ["num_azimuthal", 1],
                                  ["azimuthal_angles", 0.],
                                  ["input_temp", 0],
                                  ["input_wind", 0],
                                  ["input_humidity", 0],
                                  ["heading_angle", 0.],
                                  ["climb_angle", 0.],
                                  ["output_format", 1],
                                  ["input_xdim", 0],
                                  ["acceleration", 0.],
                                  ["turn_rate", 0.],
                                  ["pitch_rate", 0.]])

        return parameters

    def set(self, **inputs):
        """Sets case parameters.

        Any of the case parameters can be set via this function by passing
        them in as keyword arguments.

        These keywords along with the default values are listed below.

        "signature": None
        "signature_filename": "signature.data"
        "mach_number": None
        "altitude": None
        "propagation_start": None
        "propagation_points": 10000
        "padding_points": 1200
        "nonlinear": True
        "thermoviscous": True
        "relaxation": True
        "step_size": 0.01
        "reflection_factor": 1.9
        "resampled_points": 1000
        "padding_tolerance": 0.001
        "input_tolerance": 1e-6
        "input_format": 0
        "altitude_stop": None
        "num_azimuthal": 1
        "azimuthal_angles": 0.
        "input_temp": 0
        "input_wind": 0
        "input_humidity": 0
        "heading_angle": 0.
        "climb_angle": 0.
        "output_format": 1
        "input_xdim": 0
        "acceleration": 0.
        "turn_rate": 0.
        "pitch_rate": 0.

        """
        for name, value in inputs.items():
            if name in self._parameters:
                self._parameters[name] = value
            else:
                raise RuntimeError(name+" keyword argument not recognized")

    def run(self, overwrite=True):
        """Generates Panair inputfile and runs case.

        Returns
        -------
        results : dict

        Examples
        --------
        results = basic_test_case.run()
        a_weighted = results["signal_0"]["A_weighted"]
        c_weighted = results["signal_0"]["C_weighted"]

        """
        if overwrite:
            time.sleep(0.2)
            self._create_dir()
            self._write_inputfile()
            self._call_executable()

        self._parse_outputfile()

        return self._results
        # if self._check_if_successful():
        #     self._gather_results()
        #     return self._results
        # else:
        #     raise RuntimeError("Run not successful. Check panair.out for cause")

    def _create_dir(self):
        no_success = True
        while no_success:
            try:
                if os.path.exists(self._directory):
                    shutil.rmtree(self._directory)

                # create directory for case
                os.makedirs(self._directory)

                # copy in panair.exec
                executable = os.path.join(os.path.dirname(__file__), '..', self._sboom_exec)
                if os.path.isfile(executable):
                    shutil.copy2(executable, self._directory)
                else:
                    raise RuntimeError("sboom executable not found")
                no_success = False
            except:
                no_success = True

    def _write_inputfile(self, input_source = None):
        sig_filename = self._directory+self._parameters["signature_filename"]
        input_filename = self._directory+"presb.input"

        with open(sig_filename, 'w') as f:
            self._write_signature_file(f)
        if input_source is not None:
            shutil.copyfile(input_source, self._directory)
        else:
            with open(input_filename, 'w') as f:
                self._write_parameter_file(f)

    def _write_signature_file(self, f):
        num_angles = self._parameters["num_azimuthal"]
        angles = self._parameters["azimuthal_angles"]
        signatures = self._parameters["signature"]

        f.write("Number of signatures = "+str(num_angles)+"\n")
        if num_angles == 1:
            f.write("Signatures at phi="+str(angles)+"\n")
            f.write("Number of points = "+str(len(signatures))+"\n")
            for point in signatures:
                f.write("{0[0]:>12f}{0[1]:>12f}\n".format(point))

        elif num_angles > 1:
            f.write("Signatures at ")
            for angle in angles:
                f.write("phi="+angles+", ")
            f.write("\n")
            for signature in signatures:
                f.write("Number of points = "+len(signatures)+"\n")
                for point in signature:
                    f.write("{0[0]:>12f}{0[0]:>12f}\n".format(point))
        else:
            raise RuntimeError("must have positive number of azimuthal angles")

    def _write_parameter_file(self, f):
        for name, option in self._parameters.items():
            if name == "signature":
                pass
            else:
                if name in ["input_temp", "input_wind", "input_humidity"] and option != 0:
                    f.write('{0:<10}'.format(1)+"   "+name+"\n")
                    if name == "input_wind":
                        # For X wind
                        f.write('{}'.format(len(option))+"\n")
                        for i in range(len(option)):
                            f.write('{0:<8}{1}'.format(option[i][0], option[i][1])+"\n")
                        # For Y wind
                        f.write('{}'.format(len(option))+"\n")
                        for i in range(len(option)):
                            f.write('{0:<8}{1}'.format(option[i][0], option[i][2])+"\n")
                    else:
                        f.write('{}'.format(len(option))+"\n")
                        for i in range(len(option)):
                            f.write('{0:<8}{1}'.format(option[i][0], option[i][1])+"\n")
                else:
                    f.write('{0:<10}'.format(option)+"   "+name+"\n")
            # elif name == "signature_filename":
            #     f.write(self._format_opt(option)+"   "+name+"\n")
            # else:
            #     try:
            #         for i, val in enumerate(option):
            #             # treating each input as a list will allow for the
            #             # inputing of temp profiles, etc... in the future
            #             if i == 0:
            #                 f.write(self._format_opt(val)+"   "+name+"\n")
            #             else:
            #                 f.write(self._format_opt(val))
            #     except TypeError:
            #         f.write(self._format_opt(val)+"   "+name+"\n")

    def _call_executable(self):
        if sys.platform.startswith("win"):
            SEM_NOGPFAULTERRORBOX = 0x0002
            ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
            CREATE_NO_WINDOW = 0x08000000
            subprocess_flags = CREATE_NO_WINDOW
        else:
            subprocess_flags = 0
        while not os.path.isfile(self._directory+"loud.dat"):
            p = subprocess.call(os.path.join(
                self._sboom_loc, self._sboom_exec), cwd=self._directory)
            # p = subprocess.Popen(os.path.join(self._directory,
            #                                   self._sboom_exec),
            #                      stdin=subprocess.PIPE,
            #                      stdout=subprocess.PIPE,
            #                      stderr=subprocess.PIPE,
            #                      creationflags=subprocess_flags,
            #                      cwd=self._directory)
            #
            # current_time = 0
            # start = time.time()
            # total_time = 10
            # dt = .1
            # while current_time < total_time and p.poll() is None:
            #     current_time = time.time() - start
            #     time.sleep(dt)
            #
            # if current_time >= total_time:
            #     os.system("taskkill /im WerFault.exe")
            # p.kill()

    def _parse_outputfile(self):
        num_signals = self._parameters["num_azimuthal"]

        # read in loudness data
        with open(self._directory+"loud.dat") as f:
            for i in range(num_signals):
                line_items = f.readline().split()
                angle = float(line_items[6].strip(','))
                line_items = f.readline().split()
                a_weighted = float(line_items[2].strip(','))
                line_items = f.readline().split()
                c_weighted = float(line_items[2].strip(','))
                line = f.readline()
                self._results["signal_"+str(i)] = {"angle": angle,
                                                   "A_weighted": a_weighted,
                                                   "C_weighted": c_weighted}

        # read in ground signatures ***Only works for 1 signal***
        ground_sig = np.genfromtxt(self._directory+"SBground.sig",
                                   skip_header=3)

        self._results["signal_0"]["ground_sig"] = ground_sig

# rapidboom

This package provides a set of tools for rapid prediction of sonic boom
loudness on the ground from a parametric geometry description. Currently,
these tools are intended to be used in conjuction with the panairwrapper code
which performs the aerodynamic calculations and the PyLdB code for calculating
the perceived loudness.

The primary purpose of these tools is to automate the entire process of
obtaining the sonic boom loudness on the ground so that it can easily
be integrated into multidisciplinary design and optimization codes. 

The following code demonstrates how this package might be used in a 
Python script. This specific example demonstrates adding a parametric
bump to an existing axisymmetric geometry to determine how it will affect
perceived loudness at the ground.

```python
from rapidboom.sboomwrapper import SboomWrapper
import pyldb
import rapidboom.parametricgeometry as pg
import panairwrapper

import numpy as np
import matplotlib.pyplot as plt


# folder where all case files for the tools will be stored
CASE_DIR = "./axie_bump/"

REF_LENGTH = 32.92
MACH = 1.6
gamma = 1.4
R_over_L = 1

# INITIALIZE MODELS/TOOLS OF THE CASE AND SET ANY CONSTANT PARAMETERS
# import AXIE geometry from file
geometry = np.genfromtxt(CASE_DIR+"axie-geom-v1-mm.dat")
x_geom = geometry[:, 0]
r_geom = geometry[:, 1]
x_geom *= 0.001  # meters
r_geom *= 0.001  # meters

# initialize Panair
panair = panairwrapper.PanairWrapper('axie', CASE_DIR, 
                                     exe='panair')
panair.set_aero_state(MACH)
panair.set_sensor(MACH, R_over_L, REF_LENGTH)
panair.set_symmetry(1, 1)
panair.add_network('axie_surface', None)

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
    distance_along_sensor = offbody_data[:, 2]
    dp_over_p = 0.5*gamma*MACH**2*offbody_data[:, -2]
    nearfield_sig = np.array([distance_along_sensor, dp_over_p]).T
    # plt.plot(nearfield_sig[:, 0], nearfield_sig[:, 1])
    # plt.show()

    # update sBOOM settings and run
    sboom.set(signature=nearfield_sig)
    sboom_results = sboom.run()
    ground_sig = sboom_results["signal_0"]["ground_sig"]

    # grab the loudness level
    loudness_level = pyldb.perceivedloudness(ground_sig[:, 0], ground_sig[:, 1])

    return loudness_level
```

This case specific case has also been included in the uli_cases module and can
be imported and ran by simply using the code below.

```python
from rapidboom import AxieBump

bump_case = AxieBump(case_dir='./', panair_exec='panair.exe', sboom_exec='sboom.dat.allow')
loudness = bump_case.run([.1, 20., 6.])

print(loudness)
```

## Notes

sBOOM is NASA software with it's distribution controlled by them. The sBOOM
code is not included in this package and must be requested from their website
[https://software.nasa.gov/](https://software.nasa.gov/). The sboomwrapper module
included in this package is simply a programmatic interface to sBOOM and requires
that the user already has sBOOM on their system.

## Documentation

See doc strings in code. 

## Installation

Run either of the following commands in the main rapidboom folder.

'pip install .'
or
'python setup.py install'

If developing, instead use

'pip install -e .'
or
'python setup.py develop'

It is recommended that pip is used instead of invoking setup.py directly.

Additionaly, the sBOOM executable needs to be copied into the main rapidboom
folder. The name of the executable may vary by system and its name on your
system must be specified when initializing the SboomWrapper class. 

### Prerequisites

The panairwrapper package must already be installed which can be found at
[https://github.com/usuaero/panairwrapper](https://github.com/usuaero/panairwrapper).

### Getting the Source Code

The source code can be found at [https://github.com/usuaero/rapidboom](https://github.com/usuaero/rapidboom)

You can either download the source as a ZIP file and extract the contents, or 
clone the panairwrapper repository using Git. If your system does not already have a 
version of Git installed, you will not be able to use this second option unless 
you first download and install Git. If you are unsure, you can check by typing 
`git --version` into a command prompt.

#### Downloading source as a ZIP file

1. Open a web browser and navigate to [https://github.com/usuaero/rapidboom](https://github.com/usuaero/rapidboom)
2. Make sure the Branch is set to `Master`
3. Click the `Clone or download` button
4. Select `Download ZIP`
5. Extract the downloaded ZIP file to a local directory on your machine

#### Cloning the Github repository

1. From the command prompt navigate to the directory where MachUp will be installed
2. `git clone https://github.com/usuaero/panairwrapper`

## Testing
Unit tests are implemented using the pytest module and are run using the following command.

'python3 -m pytest test/'

Note: The tests have been written to assume that the executable name for sBOOM is sboom_linux for Linux and 
sboom_windows.dat.allow for Windows.

## Support
Contact doug.hunsaker@usu.edu with any questions.

## License
This project is licensed under the MIT license. See LICENSE file for more information. 

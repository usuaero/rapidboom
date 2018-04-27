"""Tests the sample cases."""

import pytest
from rapidboom import AxieBump
import platform


def test_axie_bump_case():
    CASE_DIR = "./test/sample_use_cases/axie_bump/"

    print(platform.system())
    if platform.system() == 'Linux':
        PANAIR_EXE = 'panair'
        SBOOM_EXE = 'sboom_linux'
    elif platform.system() == 'Windows':
        PANAIR_EXE = 'panair.exe'
        SBOOM_EXE = 'sboom_windows.dat.allow'
    else:
        raise RuntimeError("platfrom not recognized")

    axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE)
    axiebump.MESH_COARSEN_TOL = 0.00005
    loudness = axiebump.run([0.1, 20., 6.])

    print("C-weighted loudness", loudness)
    assert(loudness - 93.54010125368868 < 0.01)

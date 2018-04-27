"""Tests the sample cases."""

import pytest
from rapidboom import AxieBump


def test_axie_bump_case():
    CASE_DIR = "./test/sample_use_cases/axie_bump/"
    axiebump = AxieBump(CASE_DIR)
    axiebump.MESH_COARSEN_TOL = 0.00005
    loudness = axiebump.run([0.1, 20., 6.])

    print("C-weighted loudness", loudness)
    assert(loudness - 93.54010125368868 < 0.01)

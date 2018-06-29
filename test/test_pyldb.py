#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyldb unit testing
"""

import pytest
import rapidboom.pyldb as pyldb
import numpy as np


def test_PLDB():
    data = np.genfromtxt("./test/testfiles/panair_r1.sig", skip_header=3)
    time = data[:, 0]
    pressure = data[:, 1]
    PLdB = pyldb.perceivedloudness(time, pressure, pad_front=6, pad_rear=6,
                                   len_window=800)
    assert np.allclose(PLdB, 77.67985293502309, rtol=0.0, atol=10e-12)

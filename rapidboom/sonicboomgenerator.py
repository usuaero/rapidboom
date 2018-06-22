#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sonic Boom Ground Signature Generator
"""

import numpy as np
import matplotlib.pylot as plt
import pyldb

def sonic_boom_gen(sig_len, max_op, rise_time=0.01, num_points=10000, n_wave=False, shaped=False):
    if n_wave == True:
        
def _n_wave_generator(time, overpressure, rise_time, N):
    n_wave = np.zeros(time)
    time_stamp = np.linspace(0, time, num=N, endpoint=True)
    rise_len = np.nonzero(rise_time <= time_stamp)[0]

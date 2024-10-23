#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control methods.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley.
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np

from ship_ice_planner.controller.gnc import Rzyx, ssa


# MIMO nonlinear PID pole placement
def DPpolePlacement(
        e_int, M3, D3, eta3, nu3, x_d, y_d, psi_d, wn, zeta, sampleTime
):
    # PID gains based on pole placement
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))

    Kp = wn @ wn @ M3_diag
    Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
    Ki = (1.0 / 10.0) * wn @ Kp

    # DP control law - setpoint regulation
    e = eta3 - np.array([x_d, y_d, psi_d])
    e[2] = ssa(e[2])
    R = Rzyx(0.0, 0.0, eta3[2])
    tau = (  # compute the control forces/moments
            - np.matmul((R.T @ Kp), e)
            - np.matmul(Kd, nu3)
            # - np.matmul((R.T @ Ki), e_int)  # commenting out integral term
    )

    # Integral error, Euler's method
    e_int += sampleTime * e

    return tau, e_int

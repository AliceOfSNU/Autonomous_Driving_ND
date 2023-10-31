# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dt = params.dt
        self.q = params.q

        #init F
        self.F = np.identity(6)
        for i in range(3):
            self.F[i, 3+i] = self.dt

        #init Q
        third = self.q*self.dt**3/3 * np.identity(3)
        second = self.q*self.dt**2/2 * np.identity(3)
        first = self.q*self.dt * np.identity(3)
        self.Q = np.block([[third, second], 
                           [second, first]])

    def F(self):
        # TODO Step 1: implement and return system matrix F
        return self.F

    def Q(self):
        # TODO Step 1: implement and return process noise covariance Q
        return self.Q        

    def predict(self, track):
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        p_pred = self.F @ track.P @ self.F.transpose() + self.Q
        x_pred = self.F @ track.x
        track.set_x(x_pred)
        track.set_P(p_pred)

    def update(self, track, meas):
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        residual = self.gamma(track, meas)
        H = meas.sensor.get_H(track.x)
        S = self.S(track, meas, H)
        K = track.P @ H.transpose() @ np.linalg.inv(S)
        track.set_x(track.x + K@residual)
        track.set_P((np.identity(6) - K@H) @ track.P)
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        # TODO Step 1: calculate and return residual gamma
        # x is in veh space(track), z is in sensor space(meas)
        return meas.z - meas.sensor.get_hx(track.x[0:3])


    def S(self, track, meas, H):
        # TODO Step 1: calculate and return covariance of residual S
        S = H@track.P@H.transpose() + meas.R
        return S

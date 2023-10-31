# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        
        self.association_matrix = np.matrix([]) # reset matrix
        self.unassigned_meas = [idx for idx in range(len(meas_list))]
        self.unassigned_tracks = [idx for idx in range(len(track_list))]

        if len(meas_list) > 0 and len(track_list) > 0: 
            self.association_matrix = np.inf * np.ones((len(track_list), len(meas_list)))
            for track in range(len(track_list)):
                for meas in range(len(meas_list)):
                    MHD = self.MHD(track_list[track], meas_list[meas], KF)
                    if self.gating(MHD, meas_list[meas].sensor):
                        self.association_matrix[track, meas] = MHD
        
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        (track_i, meas_j) = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        # the following only works for at most one track and one measurement
        update_track = self.unassigned_tracks[track_i]
        update_meas = self.unassigned_meas[meas_j]
        
        dist = self.association_matrix[track_i, meas_j]
        if dist == np.inf:
            return np.NaN, np.NaN #minimum if infinity - nothing to associate
        
        # remove from list
        self.unassigned_tracks.pop(track_i) 
        self.unassigned_meas.pop(meas_j)
        self.association_matrix = np.delete(self.association_matrix, track_i, 0)
        self.association_matrix = np.delete(self.association_matrix, meas_j, 1)

        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        return MHD < chi2.ppf(0.995, df=3)
    

    def MHD(self, track, meas, KF):
        # TODO Step 3: calculate and return Mahalanobis distance
        residual = KF.gamma(track, meas)
        H = meas.sensor.get_H(track.x)
        S = KF.S(track, meas, H)
        return (residual.transpose() @ np.linalg.inv(S) @ residual)

    
    def associate_and_update(self, manager, meas_list, KF, sensor):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                # all entries are INF, for example.
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list, sensor)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)
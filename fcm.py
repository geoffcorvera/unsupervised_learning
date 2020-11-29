import numpy as np

class FCM(object):

    def __init__(self,c,m,X):
        # initialize memberships XXX each row should sum to 1
        memberships = None

        self.c = c
        self.m = m
        self.memberships = memberships
    
    def nextCentroid(self):
        pass

    def nextMemberships(self):
        pass

    def fit(self, X):
        pass

    def classify(self, X):
        pass
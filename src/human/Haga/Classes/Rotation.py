import numpy as np
from Classes.Euler import Euler

class Rotation():

    @staticmethod
    def to_euler(R, seq="XYZ", to_degree=True):

        if seq == "XYZ":
            beta = np.arctan2(R[0, 2], np.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2))
            alpha = np.arctan2(-R[1, 2] / np.cos(beta), R[2, 2] / np.cos(beta))
            gamma = np.arctan2(-R[0, 1] / np.cos(beta), R[0, 0] / np.cos(beta))

        elif seq == "XZY":
            beta = np.arctan2(-R[0, 1], np.sqrt(R[0, 0] ** 2 + R[0, 2] ** 2))
            alpha = np.arctan2(R[2, 1] / np.cos(beta), R[1, 1] / np.cos(beta))
            gamma = np.arctan2(R[0, 2] / np.cos(beta), R[0, 0] / np.cos(beta))

        elif seq == "ZXY":
            beta = np.arctan2(R[2, 1], np.sqrt(R[2, 0] ** 2 + R[2, 2] ** 2))
            alpha = np.arctan2(-R[0, 1] / np.cos(beta), R[1, 1] / np.cos(beta))
            gamma = np.arctan2(-R[2, 0] / np.cos(beta), R[2, 2] / np.cos(beta))

        elif seq == "ZYX":
            beta = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
            alpha = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
            gamma = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))

        eulers = np.array([alpha, beta, gamma])

        if to_degree:
            eulers = Euler.rad2deg(eulers)

        return eulers

    @staticmethod
    def relative_R(parent, child):

        R = np.dot(parent, np.linalg.inv(child))

        return R

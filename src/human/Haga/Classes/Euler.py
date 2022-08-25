import numpy as np

class Euler():

    @staticmethod
    def rad2deg(rad):
        """
        Convert angles vector from rad top deg
        Parameters:
            rad (list or ndarray):
        Returns:
            ndarray:
        """
        if isinstance(rad, list):
            rad = np.array(rad)
        else:
            rad = rad

        deg = rad / np.pi * 180

        return deg

    @staticmethod
    def deg2rad(deg):
        """
        Convert angles vector from deg top rad
        Parameters:
            deg - list or ndarray:
        Returns:
            rad - ndarray:
        """
        if isinstance(deg, list):
            deg = np.array(deg)
        else:
            deg = deg

        rad = deg / 180 * np.pi

        return rad

    @staticmethod
    def to_rotation_matrix(eulers, seq="XYZ", degrees=True):

        if degrees:
            eulers = Euler.deg2rad(eulers)

        if seq == "XYZ":
            R = Euler.rotationX(eulers[0]) @ Euler.rotationY(eulers[1]) @ Euler.rotationZ(eulers[2])

        elif seq == "XZY":
            R = Euler.rotationX(eulers[0]) @ Euler.rotationZ(eulers[1]) @ Euler.rotationY(eulers[2])

        elif seq == "ZXY":
            R = Euler.rotationZ(eulers[0]) @ Euler.rotationX(eulers[1]) @ Euler.rotationY(eulers[2])

        elif seq == "ZYX":
            R = Euler.rotationZ(eulers[0]) @ Euler.rotationY(eulers[1]) @ Euler.rotationX(eulers[2])

        return R

    @staticmethod
    def rotationX(angle):

        R = np.array([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]])

        return R

    @staticmethod
    def rotationY(angle):

        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])

        return R

    @staticmethod
    def rotationZ(angle):

        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])

        return R

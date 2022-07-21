import numpy as np
import math

class Convertion_angles_rotation_matrix():

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

        return rad / np.pi * 180

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
    def convert_q_to_euler_angles(q, sequence='ZYX', to_degree=True):

        """
        Convert quaternion vector to euler angles

        Parameters:
            q - ndarray (dim: 4,1): quaternion vector
            to_degree - bool:

        Returns:
            y - ndarray (dim: 3,1):
        """

        R = Convertion_angles_rotation_matrix.convert_q_to_R(q)
        eulers_deg = Convertion_angles_rotation_matrix.convert_R_to_euler_angles(R, sequence, to_degree)

        return eulers_deg

        # test = -R[2, 0]
        #
        # q = q.squeeze()
        #
        # if test > 0.99999:
        #
        #     a = 2 * (q[1] * q[2] - q[0] * q[3])
        #     b = 2 * (q[1] * q[3] + q[0] * q[2])
        #     roll = np.arctan2(a, b)
        #
        #     pitch = np.pi / 2
        #     yaw = 0
        #
        # elif test < -0.99999:
        #
        #     a = -2 * (q[1] * q[2] - q[0] * q[3])
        #     b = -2 * (q[1] * q[3] + q[0] * q[2])
        #     roll = np.arctan2(a, b)
        #
        #     pitch = -np.pi / 2
        #     yaw = 0
        #
        # else:
        #
        #     # a = 2 * (q[0] * q[1] + q[2] * q[3])
        #     # b = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        #     a = 2 * (q[2] * q[3] + q[0] * q[1])
        #     b = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
        #     roll = np.arctan2(a, b)
        #
        #     a = 2 * (q[1] * q[3] - q[0] * q[2])
        #     pitch = np.arcsin(-a)
        #
        #     # a = 2 * (q[0] * q[3] + q[1] * q[2])
        #     # b = 1 - 2 * (q[2] ** 2 + q[3] ** 2)
        #     a = 2 * (q[1] * q[2] + q[0] * q[3])
        #     b = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
        #     yaw = np.arctan2(a, b)
        #
        # eulers_rad = np.array([[roll], [pitch], [yaw]])
        # eulers_deg = Convertion_angles_rotation_matrix.rad2deg(eulers_rad)
        #
        # if to_degree:
        #     return eulers_deg
        # else:
        #     return eulers_rad


    @staticmethod
    def convert_R_to_euler_angles(R, sequence='ZYX', to_degree=True):
        """
        Convert rotation matrix to euler angles

        Parameters:
            R - ndarray (dim: 3,3): rotation matrix
            to_degree - bool:

        Returns:
            y - ndarray (dim: 3,1):
        """

        if sequence == 'XZY':
            test = -R[0,1]

            if test > 0.99999:
                roll = np.arcsin(R[2, 0])
                pitch = np.pi / 2
                yaw = 0
            elif test < -0.99999:
                roll = np.arcsin(-R[2, 0])
                pitch = -np.pi / 2
                yaw = 0
            else:
                roll = np.arctan2(R[2, 1], R[1, 1])
                pitch = np.arcsin(-R[0, 1])
                yaw = np.arctan2(R[0, 2], R[0, 0])

        elif sequence == 'XYZ':
            test = R[0,2]

            if test > 0.99999:
                roll = np.arcsin(R[2, 1])
                pitch = np.pi / 2
                yaw = 0
            elif test < -0.99999:
                roll = np.arcsin(R[2, 1])
                pitch = -np.pi / 2
                yaw = 0
            else:
                roll = np.arctan2(-R[1, 2], R[2, 2])
                pitch = np.arcsin(R[0, 2])
                yaw = np.arctan2(-R[0, 1], R[0, 0])

        elif sequence == 'YXZ':
            test = -R[1,2]

            if test > 0.99999:
                roll = np.arcsin(R[0, 1])
                pitch = np.pi / 2
                yaw = 0
            elif test < -0.99999:
                roll = np.arcsin(-R[0, 1])
                pitch = -np.pi / 2
                yaw = 0
            else:
                roll = np.arctan2(R[0, 2], R[2, 2])
                pitch = np.arcsin(-R[1, 2])
                yaw = np.arctan2(R[1, 0], R[1, 1])

        elif sequence == 'YZX':
            test = R[1,0]

            if test > 0.99999:
                roll = np.arcsin(R[0,2])
                pitch = np.pi / 2
                yaw = 0
            elif test < -0.99999:
                roll = np.arcsin(R[0, 2])
                pitch = -np.pi / 2
                yaw = 0
            else:
                roll = np.arctan2(-R[2, 0], R[0, 0])
                pitch = np.arcsin(R[1, 0])
                yaw = np.arctan2(-R[1, 2], R[1, 1])

        elif sequence == 'ZYX':
            test = -R[2, 0]

            if test > 0.99999:
                roll = np.arctan2(R[0, 1], R[0, 2])
                pitch = np.pi / 2
                yaw = 0
            elif test < -0.99999:
                roll = np.arctan2(-R[0, 1], -R[0, 2])
                pitch = -np.pi / 2
                yaw = 0
            else:
                roll = np.arctan2(R[1, 0], R[0, 0])
                pitch = np.arcsin(-R[2, 0])
                yaw = np.arctan2(R[2, 1], R[2, 2])

        else: #ZXY
            test = R[2,1]

            if test > 0.99999:
                roll = np.arctan2(R[1, 0], R[0, 0])
                pitch = np.pi / 2
                yaw = 0
            elif test < -0.99999:
                roll = np.arctan2(R[1, 0], R[0, 0])
                pitch = -np.pi / 2
                yaw = 0
            else:
                roll = np.arctan2(-R[0, 1], R[1, 1])
                pitch = np.arcsin(R[2, 1])
                yaw = np.arctan2(-R[2, 0], R[2, 2])

        eulers_rad = np.array([[roll], [pitch], [yaw]])
        eulers_deg = Convertion_angles_rotation_matrix.rad2deg(eulers_rad)

        if to_degree:
            return eulers_deg
        else:
            return eulers_rad

    @staticmethod
    def convert_q_to_R(q):
        """
        Convert quaternion vector to rotation matrix

        Parameters:
            q - ndarray (dim: 4,1): quaternion vector

        Returns:
            R - ndarray (dim: 3,3): rotation matrix
        """

        c00 = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
        c01 = 2 * (q[1] * q[2] - q[0] * q[3])
        c02 = 2 * (q[1] * q[3] + q[0] * q[2])

        c10 = 2 * (q[1] * q[2] + q[0] * q[3])
        c11 = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
        c12 = 2 * (q[2] * q[3] - q[0] * q[1])

        c20 = 2 * (q[1] * q[3] - q[0] * q[2])
        c21 = 2 * (q[2] * q[3] + q[0] * q[1])
        c22 = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

        X = np.array([c00, c10, c20])
        X = X / np.linalg.norm(X)  # Normalize vector

        Y = np.array([c01, c11, c21])
        Y = Y / np.linalg.norm(Y)  # Normalize vector

        Z = np.array([c02, c12, c22])
        Z = Z / np.linalg.norm(Z)  # Normalize vector

        R = np.concatenate([X, Y, Z], axis=1)

        return R

    @staticmethod
    def convert_euler_angles_to_R(psi, theta, phi, sequence='ZYX'):
        """
        Euler angle →　Rotation matrix

        :param psi: [deg]
        :param theta: [deg]
        :param phi: [deg]
        :return: rotation matrix
        """

        psi = Convertion_angles_rotation_matrix.deg2rad(psi)
        theta = Convertion_angles_rotation_matrix.deg2rad(theta)
        phi = Convertion_angles_rotation_matrix.deg2rad(phi)

        if sequence == 'XZX':
            R_1 = np.array([[1, 0, 0],
                              [0, np.cos(psi), -np.sin(psi)],
                              [0, np.sin(psi), np.cos(psi)]])
            R_2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])
            R_3 = np.array([[1, 0, 0],
                              [0, np.cos(phi), -np.sin(phi)],
                              [0, np.sin(phi), np.cos(phi)]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'XYX':
            R_1 = np.array([[1, 0, 0],
                            [0, np.cos(psi), -np.sin(psi)],
                            [0, np.sin(psi), np.cos(psi)]])
            R_2 = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
            R_3 = np.array([[1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'YXY':
            R_1 = np.array([[np.cos(psi), 0, np.sin(psi)],
                            [0, 1, 0],
                            [-np.sin(psi), 0, np.cos(psi)]])
            R_2 = np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
            R_3 = np.array([[np.cos(phi), 0, np.sin(phi)],
                            [0, 1, 0],
                            [-np.sin(phi), 0, np.cos(phi)]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'YZY':
            R_1 = np.array([[np.cos(psi), 0, np.sin(psi)],
                            [0, 1, 0],
                            [-np.sin(psi), 0, np.cos(psi)]])
            R_2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            R_3 = np.array([[np.cos(phi), 0, np.sin(phi)],
                            [0, 1, 0],
                            [-np.sin(phi), 0, np.cos(phi)]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'ZYZ':
            R_1 = np.array([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1]])
            R_2 = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
            R_3 = np.array([[np.cos(phi), -np.sin(phi), 0],
                            [np.sin(phi), np.cos(phi), 0],
                            [0, 0, 1]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'ZXZ':
            R_1 = np.array([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1]])
            R_2 = np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
            R_3 = np.array([[np.cos(phi), -np.sin(phi), 0],
                            [np.sin(phi), np.cos(phi), 0],
                            [0, 0, 1]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'XZY':
            R_1 = np.array([[1, 0, 0],
                            [0, np.cos(psi), -np.sin(psi)],
                            [0, np.sin(psi), np.cos(psi)]])
            R_2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])
            R_3 = np.array([[np.cos(phi), 0, np.sin(phi)],
                                [0, 1, 0],
                                [-np.sin(phi), 0, np.cos(phi)]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'XYZ':
            R_1 = np.array([[1, 0, 0],
                            [0, np.cos(psi), -np.sin(psi)],
                            [0, np.sin(psi), np.cos(psi)]])
            R_2 = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
            R_3 = np.array([[np.cos(phi), -np.sin(phi), 0],
                            [np.sin(phi), np.cos(phi), 0],
                            [0, 0, 1]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'YXZ':
            R_1 = np.array([[np.cos(psi), 0, np.sin(psi)],
                            [0, 1, 0],
                            [-np.sin(psi), 0, np.cos(psi)]])
            R_2 = np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
            R_3 = np.array([[np.cos(phi), -np.sin(phi), 0],
                            [np.sin(phi), np.cos(phi), 0],
                            [0, 0, 1]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'YZX':
            R_1 = np.array([[np.cos(psi), 0, np.sin(psi)],
                            [0, 1, 0],
                            [-np.sin(psi), 0, np.cos(psi)]])
            R_2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            R_3 = np.array([[1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        elif sequence == 'ZYX':
            R_1 = np.array([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1]])
            R_2 = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
            R_3 = np.array([[1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        else: #ZXY
            R_1 = np.array([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1]])
            R_2 = np.array([[1, 0, 0],
                              [0, np.cos(theta), -np.sin(theta)],
                              [0, np.sin(theta), np.cos(theta)]])
            R_3 = np.array([[np.cos(phi), 0, np.sin(phi)],
                                [0, 1, 0],
                                [-np.sin(phi), 0, np.cos(phi)]])

            R = np.dot(np.dot(R_1, R_2), R_3)

        return R

class Quaternion():

    def __init__(self):
        """
        Quaternion based approach to compute euler angle using angular velocity
        """

        super().__init__()

    def initial(self):

        self.q = np.array([[1], [0], [0], [0]])  # Initial state of the quaternion

        self.R = Convertion_angles_rotation_matrix.convert_q_to_R(self.q)
        self.eulers_rad = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(self.q, to_degree=False)
        self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

    def update(self, w, dt=0.01):
        """
        Rotate quaternion (q) given angular velocity (degree/s) (w)

        Parameters:
            w - ndarray (dim: 3,1): angular velocity vector
            dt - float:
        """

        if isinstance(w, list):
            w = np.array(w)

        if len(w.shape) == 1:
            w = w[:, np.newaxis]

        if w.shape[0] != 3:
            print("w does not have three components")
        else:
            Sq = np.array([[-self.q[1], -self.q[2], -self.q[3]],
                           [self.q[0], -self.q[3], self.q[2]],
                           [self.q[3], self.q[0], -self.q[1]],
                           [-self.q[2], self.q[1], self.q[0]]]).squeeze()

            self.q = np.dot(dt / 2 * Sq, w) + self.q

            self.R = Convertion_angles_rotation_matrix.convert_q_to_R(self.q)
            self.eulers_rad = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(self.q, to_degree=False)
            self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

    def reset_angle(self):
        """
        Reset quaternion (q) to [1, 0, 0, 0]
        Reset eulers_rad and eulers_deg to [0, 0, 0]
        """
        self.q = np.array([[1], [0], [0], [0]])  # Initial state of the quaternion
        self.eulers_rad = np.array([[0], [0], [0]])
        self.eulers_deg = np.array([[0], [0], [0]])

class Accelerometer():

    """
    Calculate euler angle using calibrated accelerometer data
    """

    def __init__(self):

        super().__init__()

    def initial(self, a):
        """
        Parameters:
            a - ndarray (dim: 3,1): Initial acceleration [m/sec^2]
            to_degree - bool:
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if a.shape[0] != 3:
            print("a does not have three components")
        else:
            self.eulers_rad_initial = self.process(a)
            self.eulers_deg_initial = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad_initial)

    def process(self, a):
        """
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
        Returns:
            y - ndarray (dim: 3,1): Eulers angles [rad]
        """

        roll = math.atan2(a[1][0], math.sqrt(a[0][0] ** 2.0 + a[2][0] ** 2.0))
        pitch = math.atan2(-a[0][0], math.sqrt(a[1][0] ** 2.0 + a[2][0] ** 2.0))
        yaw = 0

        return np.array([roll, pitch, yaw])[:, np.newaxis]

    def update(self, a):
        """
        Process accelerometer data to euler angles
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if a.shape[0] != 3:
            print("a does not have three components")
        else:
            self.eulers_rad = self.process(a)
            self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

    def reset_angle(self):
        """
        Reset eulers_rad and eulers_deg to [0, 0, 0]
        """
        self.eulers_rad = np.array([[0], [0], [0]])
        self.eulers_deg = np.array([[0], [0], [0]])

class Gyroscope():
    """
    Calculate euler angle using angular velocity
    """
    def __init__(self):

        super().__init__()

    def initial(self, a):
        """
        Parameters:
            a - ndarray (dim: 3,1): Initial acceleration [m/sec^2]
        """
        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if a.shape[0] != 3:
            print("a does not have three components")
        else:

            accelerometer = Accelerometer()
            accelerometer.initial(a=a)

            self.eulers_rad_initial = accelerometer.eulers_rad_initial
            self.eulers_deg_initial = accelerometer.eulers_deg_initial

            self.eulers_rad = self.eulers_rad_initial
            self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad_initial)

    def process(self, angles, w):
        """
        Calculate the rate change of Roll & Pitch angle
        Parameters:
            angles - ndarray (dim: 3,1): Eulers angles [rad]
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
        Returns:
            y - ndarray (dim: 3,1): Rate change [rad/sec]
        """

        roll = angles[0, 0]
        pitch = angles[1, 0]

        p = w[0, 0] / 180 * np.pi
        q = w[1, 0] / 180 * np.pi
        r = w[2, 0] / 180 * np.pi

        p = w[0, 0]
        q = w[1, 0]
        r = w[2, 0]

        roll_dot = p + math.sin(roll) * math.tan(pitch) * q + math.cos(roll) * math.tan(pitch) * r
        pitch_dot = math.cos(roll) * q - math.sin(roll) * r
        yaw_dot = 0

        return np.array([roll_dot, pitch_dot, yaw_dot])[:, np.newaxis]

    def update(self, w, dt=0.01):
        """
        Process gyroscope data to euler angles
        Parameters:
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
            dt - Sampling time [sec]
        """

        if isinstance(w, list):
            w = np.array(w)

        if len(w.shape) == 1:
            w = w[:, np.newaxis]

        if w.shape[0] != 3:
            print("a does not have three components")
        else:
            self.eulers_rad_pre = self.eulers_rad.copy()

            euler_dot = self.process(self.eulers_rad_pre, w)

            roll_dot = euler_dot[0, 0]
            pitch_dot = euler_dot[1, 0]

            roll_up = self.eulers_rad_pre[0, 0] + dt * roll_dot
            pitch_up = self.eulers_rad_pre[1, 0] + dt * pitch_dot
            yaw_up = 0

            self.eulers_rad = np.array([roll_up, pitch_up, yaw_up])[:, np.newaxis]
            self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

    def reset_angle(self):
        """
        Reset eulers_rad and eulers_deg to [0, 0, 0]
        """
        self.eulers_rad = np.array([[0], [0], [0]])
        self.eulers_deg = np.array([[0], [0], [0]])

class Complimentary_Filter():
    """
    Calculate the euler angle using complimentary filter
    """
    def __init__(self):

        super().__init__()

        self.alpha = 0.1

    def initial(self, a):
        """
        Parameters:
            a - ndarray (dim: 3,1): Initial acceleration [m/sec^2]
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if a.shape[0] != 3:
            print("a does not have three components")
        else:
            self.accelerometer = Accelerometer()
            self.accelerometer.initial(a=a)

            self.gyroscope = Gyroscope()
            self.gyroscope.initial(a=a)

            self.eulers_rad_initial = self.accelerometer.eulers_rad_initial
            self.eulers_deg_initial = self.accelerometer.eulers_deg_initial

            self.eulers_rad = self.eulers_rad_initial
            self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad_initial)

    def process(self, a, w, dt, angles_hat):
        """
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
            dt - Sampling time [sec]
            angles_hat - Roll & Pitch angle [rad]
        Returns:
            y - ndarray (dim: 3,1): [rad]
        """

        hat_acc = self.accelerometer.process(a)
        dot_gyr = self.gyroscope.process(angles_hat, w)

        roll_hat = angles_hat[0, 0]
        pitch_hat = angles_hat[1, 0]

        roll_hat = (1 - self.alpha) * (roll_hat + dt * dot_gyr[0, 0]) + self.alpha * hat_acc[0, 0]
        pitch_hat = (1 - self.alpha) * (pitch_hat + dt * dot_gyr[1, 0]) + self.alpha * hat_acc[1, 0]
        yaw_hat = 0

        return np.array([roll_hat, pitch_hat, yaw_hat])[:, np.newaxis]

    def update(self, a, w, dt=0.01):
        """
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
            dt - Sampling time [sec]
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if isinstance(w, list):
            w = np.array(w)

        if len(w.shape) == 1:
            w = w[:, np.newaxis]

        if a.shape[0] != 3 or w.shape[0] != 3:
            print("a or w does not have three components")

        else:
            self.eulers_rad_pre = self.eulers_rad.copy()

            self.eulers_rad = self.process(a, w, dt, self.eulers_rad_pre)
            self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

    def reset_angle(self):
        """
        Reset eulers_rad and eulers_deg to [0, 0, 0]
        """
        self.eulers_rad = np.array([[0], [0], [0]])
        self.eulers_deg = np.array([[0], [0], [0]])

class Kalman_Filter():

    def __init__(self):

        super().__init__()

    def initial(self, dt):
        """
        Parameters:
            dt - Sampling time [sec]
        """

        # Initial states
        self.X = np.array([[0], [0], [0], [0]])

        self.A = np.array([[1, -dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dt], [0, 0, 0, 1]])
        self.B = np.array([[dt, 0], [0, 0], [0, dt], [0, 0]])
        self.C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # Predicted process covariance matrix
        self.P = np.identity(4) * 0.01

        # Process variance - Error in the process of the covariance matrix
        self.Q = np.identity(4) * 0.01

        # Observation error
        self.R = np.identity(2) * 10

        # Simple identity matrix
        self.I = np.eye(4)

        self.eulers_rad_initial = [self.X[0][0], self.X[2][0], 0]
        self.eulers_deg_initial = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad_initial)

    def process(self, u, z):
        """
        Parameters:
            u - ndarray (dim: 3,1): Eulers angle calculated by acceleration [rad]
            z - ndarray (dim: 3,1): The rate change of Roll & Pitch angle calculated by angular velocity [rad/sec]
        Returns:
            y - ndarray (dim: 3,1): Eulers angle [rad]
        """

        self.u = u[0:2]
        self.z = z[0:2]

        # Predicted states matrix (X_p = A.X + B.U + W)
        self.X_p = np.dot(self.A, self.X) + np.dot(self.B, self.u)

        # Predicted covariance matrix (P_p = A.P.A' + Q)
        self.P_p = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.Q

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        self.K = np.dot(np.dot(self.P_p, self.C.transpose()), np.linalg.inv(np.dot(np.dot(self.C, self.P_p), self.C.transpose()) + self.R))

        self.yHatBar = np.dot(self.C, self.X_p)

        # Current observations matrix (Y = C.Y + Z)
        Y = self.z

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.dot(self.K, Y - self.yHatBar)

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = np.dot(self.I - np.dot(self.K, self.C), self.P_p)

        return np.array([self.X[0][0], self.X[2][0], 0])[:, np.newaxis]

    def update(self, a, w):
        """
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if isinstance(w, list):
            w = np.array(w)

        if len(w.shape) == 1:
            w = w[:, np.newaxis]

        if a.shape[0] != 3 or w.shape[0] != 3:
            print("a or w does not have three components")
        else:
            self.accelerometer = Accelerometer()
            self.accelerometer.initial(a=a)

            self.gyroscope = Gyroscope()
            self.gyroscope.initial(a=a)

            acc_angles_hat = self.accelerometer.process(a)
            angles_hat = np.array([self.X[0][0], self.X[2][0], 0])[:, np.newaxis]
            gyr_angles_rate_hat = self.gyroscope.process(angles=angles_hat, w=w)

            self.eulers_rad = self.process(u=gyr_angles_rate_hat, z=acc_angles_hat)
            self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

    def reset_angle(self):
        """
        Reset eulers_rad and eulers_deg to [0, 0, 0]
        """

        self.eulers_rad = np.array([[0], [0], [0]])
        self.eulers_deg = np.array([[0], [0], [0]])

class Extended_kalman_filter():

    def __init__(self):

        super().__init__()

    def initial(self, dt):

        """
        Parameters:
            dt - Sampling time [sec]
        """

        self.dt = dt

        # Initial states
        q = np.array([[1], [0], [0], [0]])     # Quaternion
        b = np.array([[0], [0], [0]])          # Bias

        self.accelReference = np.array([[0], [0], [1]])
        self.magReference = np.array([[0], [0.653], [-0.758]])

        # self.mag_Ainv = np.array([[ 2.06423128e-03, -1.04778851e-04, -1.09416190e-06],
        #                           [-1.04778851e-04,  1.91693168e-03,  1.79409312e-05],
        #                           [-1.09416190e-06,  1.79409312e-05,  1.99819154e-03]])
        # self.mag_b = np.array([[80.51340236], [37.08931099], [105.6731885]])

        self.X = np.concatenate([q, b], axis=0)  # Self.X

        self.A, self.B = self.define_A_B(q=q, dt=self.dt)

        # # W (noise in the predicted state)
        # self.W = np.zeros(shape=(7, 1))

        #self.yHatBar = np.zeros(shape=(1, 3))

        # Predicted process covariance matrix
        self.P = np.identity(7) * 0.01

        # Process variance - Error in the process of the covariance matrix
        self.Q = np.identity(7) * 0.01

        # Observation error
        self.R = np.identity(6) * 0.01

        # Conversion matrix
        #self.H = self.define_H(q=q)

        # Simple identity matrix
        self.I = np.eye(7)

        # Information to keep
        self.q = self.X[0:4]
        self.eulers_rad_initial = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(q=self.q, to_degree=False)
        self.eulers_deg_initial = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad_initial)

    def define_A_B(self, q, dt):

        # A and B used in (X_p = A.X + B.U + W) to compute states matrix
        Sq = np.array([[-q[1], -q[2], -q[3]],
                       [q[0], -q[3], q[2]],
                       [q[3], q[0], -q[1]],
                       [-q[2], q[1], q[0]]]).squeeze()

        tmp1 = np.concatenate([np.identity(4), -dt / 2 * Sq], axis=1)
        tmp2 = np.concatenate([np.zeros((3, 4)), np.identity(3)], axis=1)

        A = np.concatenate([tmp1, tmp2], axis=0)
        B = np.concatenate([dt / 2 * Sq, np.zeros((3, 3))], axis=0)

        return A, B

    def normalize_quaternion(self, q):

        mag = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)**0.5

        return q / mag

    def getAccelVector(self, a):

        return a / np.linalg.norm(a)

    def getMagVector(self, m):

        # magGaussRaw = np.dot(self.mag_Ainv, np.array(m) - self.mag_b)
        # magGauss_N = np.dot(Convertion_angles_rotation_matrix.convert_q_to_R(q=self.X[0:4]), magGaussRaw)

        magGauss_N = np.dot(Convertion_angles_rotation_matrix.convert_q_to_R(q=self.X_p[0:4]), m) #Body frame →　World frame
        magGauss_N[2] = 0
        magGauss_N = magGauss_N / (magGauss_N[0] ** 2 + magGauss_N[1] ** 2) ** 0.5
        magGuass_B = np.dot(Convertion_angles_rotation_matrix.convert_q_to_R(q=self.X_p[0:4]).transpose(), magGauss_N)  #World frame → Body frame

        return magGuass_B

    def define_J(self, q, reference):

        # Jacobian matrix

        e00 = q[0] * reference[0] + q[3] * reference[1] - q[2] * reference[2]
        e01 = q[1] * reference[0] + q[2] * reference[1] + q[3] * reference[2]
        e02 = -q[2] * reference[0] + q[1] * reference[1] - q[0] * reference[2]
        e03 = -q[3] * reference[0] + q[0] * reference[1] + q[1] * reference[2]
        e10 = -q[3] * reference[0] + q[0] * reference[1] + q[1] * reference[2]
        e11 = q[2] * reference[0] - q[1] * reference[1] + q[0] * reference[2]
        e12 = q[1] * reference[0] + q[2] * reference[1] + q[3] * reference[2]
        e13 = -q[0] * reference[0] - q[3] * reference[1] + q[2] * reference[2]
        e20 = q[2] * reference[0] - q[1] * reference[1] + q[0] * reference[2]
        e21 = q[3] * reference[0] - q[0] * reference[1] - q[1] * reference[2]
        e22 = q[0] * reference[0] + q[3] * reference[1] - q[2] * reference[2]
        e23 = q[1] * reference[0] + q[2] * reference[1] + q[3] * reference[2]
        J = 2 * np.array([[e00, e01, e02, e03],
                          [e10, e11, e12, e13],
                          [e20, e21, e22, e23]]).squeeze()
        return J

    def define_H(self, q):

        hPrime_a = self.define_J(q=q, reference=self.accelReference)
        hPrime_m = self.define_J(q=q, reference=self.magReference)

        tmp1 = np.concatenate((hPrime_a, np.zeros((3, 3))), axis=1)
        tmp2 = np.concatenate((hPrime_m, np.zeros((3, 3))), axis=1)

        H = np.concatenate((tmp1, tmp2), axis=0)

        return H

    def predict_acc_mag(self, q):

        # Accel and mag
        accelBar = np.dot(Convertion_angles_rotation_matrix.convert_q_to_R(q=q).transpose(), self.accelReference)
        magBar = np.dot(Convertion_angles_rotation_matrix.convert_q_to_R(q=q).transpose(), self.magReference)

        return np.concatenate((accelBar, magBar), axis=0)

    def update(self, w, a, m, dt):

        """
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
            m - ndarray (dim: 3,1): Magnetometer [-]
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if isinstance(w, list):
            w = np.array(w)

        if len(w.shape) == 1:
            w = w[:, np.newaxis]

        if isinstance(m, list):
            m = np.array(m)

        if len(m.shape) == 1:
            m = m[:, np.newaxis]

        if a.shape[0] != 3 or w.shape[0] != 3 or m.shape[0] != 3:
            print("a, w or m does not have three components")

        self.U = w

        q = self.X[0:4]

        # Predicted states matrix (X_p = A.X + B.U + W)
        self.A, self.B = self.define_A_B(q=q, dt=dt)
        self.X_p = np.dot(self.A, self.X) + np.dot(self.B, self.U)
        self.X_p[0:4] = self.normalize_quaternion(self.X_p[0:4])

        # Predicted covariance matrix (P_p = A.P.A' + Q)
        self.P_p = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.Q

        # Conversion matrix
        self.H = self.define_H(q=self.X_p[0:4])

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        self.K = np.dot(np.dot(self.P_p, self.H.transpose()), np.linalg.inv(np.dot(np.dot(self.H, self.P_p), self.H.transpose()) + self.R))

        self.yHatBar = self.predict_acc_mag(q=self.X_p[0:4])

        # Current observations matrix (Y = C.Y + Z)
        #magGuass_B = self.getMagVector(m)
        magGuass_B = m / np.linalg.norm(m)
        accel_B = self.getAccelVector(a)
        Y = np.concatenate((accel_B, magGuass_B), axis=0)

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.dot(self.K, Y - self.yHatBar)
        self.X[0:4] = self.normalize_quaternion(self.X[0:4])

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = np.dot(self.I - np.dot(self.K, self.H), self.P_p)

        # Information to keep
        self.q = self.X[0:4]
        self.eulers_rad = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(q=self.q, to_degree=False)
        self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

    def reset_angle(self):
        """
        Reset eulers_rad and eulers_deg to [0, 0, 0]
        """

        self.eulers_rad = np.array([[0], [0], [0]])
        self.eulers_deg = np.array([[0], [0], [0]])

class Adaptive_complementary_Kalman_Filter_Sub():

    def __init__(self, dt):

        # Initial state
        self.X = np.array([[0], [0]])
        self.P = np.identity(2)
        # Process variance - Error in the process of the covariance matrix
        self.Q = np.array([[0.3, 0.3], [0.3, 0.3]])
        # Observation error
        self.R = 0.01
        self.F = np.array([[1, dt], [0, 1]])
        self.B = np.array([[dt], [0]])
        # Conversion matrix
        self.H = np.array([1, 0]).reshape([1,2])

    def update(self, u, z):

        # Predicted states matrix (X_p = F.X + B.u)
        self.X_p = np.dot(self.F, self.X) + np.dot(self.B, u)

        # Predicted covariance matrix (P_p = F.P.F' + Q)
        self.P_p = np.dot(np.dot(self.F, self.P), self.F.transpose()) + self.Q

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        self.K = np.multiply(np.dot(self.P_p, self.H.reshape(-1, 1)), 1 / (np.dot(np.dot(self.H, self.P_p), self.H.reshape(-1, 1)) + self.R))

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.multiply(self.K, (z - np.dot(self.H, self.X_p)))

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = self.P_p - np.dot(np.dot(self.K, self.H), self.P_p)

class Adaptive_Complementary_Kalman_Filter():

    def __init__(self):

        super().__init__()

    def initial(self, dt):

        """
        Parameters:
            dt - Sampling time [sec]
        """

        self.dt = dt

        self.roll_kf = Adaptive_complementary_Kalman_Filter_Sub(dt)
        self.pitch_kf = Adaptive_complementary_Kalman_Filter_Sub(dt)
        self.yaw_kf = Adaptive_complementary_Kalman_Filter_Sub(dt)

        self.eulers_rad_initial = np.array([[0.0], [0.0], [0.0]])
        self.eulers_deg_initial = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad_initial)

    def get_roll(self, a):
        roll = math.atan2(a[1], math.sqrt(a[0] ** 2.0 + a[2] ** 2.0))
        return roll

    def get_pitch(self, a):
        pitch = math.atan2(-a[0], math.sqrt(a[1] ** 2.0 + a[2] ** 2.0))
        return pitch

    def get_yaw(self, m, roll, pitch):
        h_x = m[1] * math.cos(roll) - m[2] * math.sin(roll)
        h_y = m[0] * math.cos(pitch) + m[1] * math.sin(pitch) * math.sin(roll) + m[2] * math.sin(pitch) * math.cos(roll)
        yaw = math.atan2(h_y, h_x)
        return yaw

    def update(self, w, a, m, dt):

        """
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
            m - ndarray (dim: 3,1): Magnetometer [-]
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if isinstance(w, list):
            w = np.array(w)

        if len(w.shape) == 1:
            w = w[:, np.newaxis]

        if isinstance(m, list):
            m = np.array(m)

        if len(m.shape) == 1:
            m = m[:, np.newaxis]

        if a.shape[0] != 3 or w.shape[0] != 3 or m.shape[0] != 3:
            print("a, w or m does not have three components")

        self.dt = dt

        roll = self.get_roll(a)
        self.roll_kf.update(u=w[0], z=roll)
        roll = self.roll_kf.X[0,0]

        pitch = self.get_pitch(a)
        self.pitch_kf.update(u=w[1], z=pitch)
        pitch = self.pitch_kf.X[0,0]

        yaw = self.get_yaw(m, roll, pitch)
        self.yaw_kf.update(u=w[2], z=yaw)
        yaw = self.yaw_kf.X[0, 0]

        roll_angle = Convertion_angles_rotation_matrix.rad2deg(roll)
        pitch_angle = Convertion_angles_rotation_matrix.rad2deg(pitch)
        yaw_angle = Convertion_angles_rotation_matrix.rad2deg(yaw)

        self.eulers_deg = np.array([[roll_angle], [pitch_angle], [yaw_angle]])
        self.eulers_rad = Convertion_angles_rotation_matrix.deg2rad(self.eulers_deg)

    def reset_angle(self):
        """
        Reset eulers_rad and eulers_deg to [0, 0, 0]
        """

        self.eulers_rad = np.array([[0], [0], [0]])
        self.eulers_deg = np.array([[0], [0], [0]])

class Gradient_Descent_Kalman_Filter():

    def __init__(self):

        super().__init__()

    def initial(self, dt):

        """
        Parameters:
            dt - Sampling time [sec]
        """

        self.dt = dt

        # Initial values
        self.X = np.array([[1], [0], [0], [0]])
        self.P = np.identity(4) * 2

        # the process covariance matrix
        # var = np.array([(0.5647 / 180 * np.pi) ** 2, (0.5647 / 180 * np.pi) ** 2, (0.5647 / 180 * np.pi) ** 2])
        # Q1 = [var[0] + var[1] + var[2], -var[0] + var[1] - var[2], -var[0] - var[1] + var[2], var[0] - var[2] - var[2]]
        # Q2 = [-var[0] + var[1] - var[2], var[0] + var[1] + var[2], var[0] - var[1] - var[2], -var[0] - var[2] + var[2]]
        # Q3 = [-var[0] - var[1] + var[2], var[0] - var[1] - var[2], var[0] + var[1] + var[2], -var[0] + var[2] - var[2]]
        # Q4 = [var[0] - var[1] - var[2], -var[0] - var[1] + var[2], -var[0] + var[1] - var[2], var[0] + var[2] + var[2]]
        # self.Q = np.array([[Q1], [Q2], [Q3], [Q4]])

        self.Q = np.identity(4) * 0.01

        # the transformation matrix (obsevation　→　state)
        self.H = np.identity(4)

        # the identity matrix
        self.I = np.identity(4)

        # the observations covariance matrix
        self.R = np.identity(4) * 0.01

        # Observation vector
        self.z = np.array([[1], [0], [0], [0]])

        # Information to keep
        self.q = self.X[0:4]
        self.eulers_rad_initial = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(q=self.q, to_degree=False)
        self.eulers_deg_initial = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad_initial)

    def normalize_quaternion(self, q):
        mag = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
        return q / mag

    def quaternion_product(self, a, b):
        """
        外積計算
        :param a: ndarray (4,1)
        :param b: ndarray (4,1)
        :return: ndarray (4,1)
        """

        a1 = a[0, 0]
        a2 = a[1, 0]
        a3 = a[2, 0]
        a4 = a[3, 0]

        b1 = b[0,0]
        b2 = b[1,0]
        b3 = b[2,0]
        b4 = b[3,0]

        p1 = a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4
        p2 = a1 * b2 + a2 * b1 + a3 * b4 - a4 * b3
        p3 = a1 * b3 - a2 * b4 + a3 * b1 + a4 * b2
        p4 = a1 * b4 + a2 * b3 - a3 * b2 + a4 * b1

        P = np.array([[p1], [p2], [p3], [p4]])

        return P

    def gradient_descent(self, accel, mag, q, mu):

        q1 = q[0,0]
        q2 = q[1,0]
        q3 = q[2,0]
        q4 = q[3,0]

        i = 1

        while i <= 10 :
            fg1 = 2 * (q2 * q4 - q1 * q3) - accel[0,0]
            fg2 = 2 * (q1 * q2 + q3 * q4) - accel[1,0]
            fg3 = 2 * (0.5 - q2 ** 2 - q3 ** 2) - accel[2,0]
            fg = np.array([[fg1], [fg2], [fg3]])

            Jg1 = np.array([-2 * q3, 2 * q4, -2 * q1, 2 * q2])
            Jg2 = np.array([2 * q2, 2 * q1, 2 * q4, 2 * q3])
            Jg3 = np.array([0, -4 * q2, -4 * q3, 0])
            Jg = np.array([[Jg1], [Jg2], [Jg3]]).squeeze()

            mag = mag / np.linalg.norm(mag)

            q_coniug = np.array([[q[0,0]], [-q[1,0]], [-q[2,0]], [-q[3,0]]])     #共役複素数

            hTemp = self.quaternion_product(q, np.array([[0], [mag[0,0]], [mag[1,0]], [mag[2,0]]]))
            h = self.quaternion_product(hTemp, q_coniug)

            # b = np.array([[math.sqrt(h[1,0] ** 2 + h[2,0] ** 2)], [0], [h[3,0]]])
            b = np.array([[0], [math.sqrt(h[1,0] ** 2 + h[2,0] ** 2)], [h[3,0]]])
            b = b / np.linalg.norm(b)

            #b = np.array([[0], [0.653], [-0.758]])

            # fb1 = 2 * b[0,0] * (0.5 - q3 ** 2 - q4 ** 2) + 2 * b[2,0] * (q2 * q4 - q1 * q3) - mag[0,0]
            # fb2 = 2 * b[0,0] * (q2 * q3 - q1 * q4) + 2 * b[2,0] * (q1 * q2 + q3 * q4) - mag[1,0]
            # fb3 = 2 * b[0,0] * (q1 * q3 + q2 * q4) + 2 * b[2,0] * (0.5 - q2 ** 2 - q3 ** 2) - mag[2,0]
            # fb = np.array([[fb1], [fb2], [fb3]])
            #
            # Jb1 = np.array([-2 * b[2,0] * q3, 2 * b[2,0] * q4,  -4 * b[0,0] * q3 - 2 * b[2,0] * q1, -4 * b[0,0] * q4 + 2 * b[2,0] * q2])
            # Jb2 = np.array([-2 * b[0,0] * q4 + 2 * b[2,0] * q2, 2 * b[0,0] * q3 + 2 * b[2,0] * q1, 2 * b[0,0] * q2 + 2 * b[2,0] * q4, -2 * b[0,0] * q1 + 2 * b[2,0] * q3])
            # Jb3 = np.array([2 * b[0,0] * q3, 2 * b[0,0] * q4 - 4 * b[2,0] * q2, 2 * b[0,0] * q1 - 4 * b[2,0] * q3, 2 * b[0,0] * q2])
            # Jb = np.array([[Jb1], [Jb2], [Jb3]]).squeeze()

            fb1 = 2 * b[1,0] * (q1*q4 + q2*q3) + 2 * b[2,0] * (q2 * q4 - q1 * q3) - mag[0,0]
            fb2 = 2 * b[1,0] * (0.5 - q2 **2 - q4 **2 ) + 2 * b[2,0] * (q1 * q2 + q3 * q4) - mag[1,0]
            fb3 = 2 * b[1,0] * (q3 * q4 - q1 * q2) + 2 * b[2,0] * (0.5 - q2 ** 2 - q3 ** 2) - mag[2,0]
            fb = np.array([[fb1], [fb2], [fb3]])

            Jb1 = np.array([2*b[1,0]*q4 -2 * b[2,0] * q3, 2*b[1,0]*q3 + 2 * b[2,0] * q4,  2 * b[1,0] * q2 - 2 * b[2,0] * q1, 2 * b[1,0] * q1 + 2 * b[2,0] * q2])
            Jb2 = np.array([2 * b[2,0] * q2, -4 * b[1,0] * q2 + 2 * b[2,0] * q1, 2 * b[2,0] * q4, -4 * b[1,0] * q4 + 2 * b[2,0] * q3])
            Jb3 = np.array([-2 * b[1,0] * q2, -2 * b[1,0] * q1 - 4 * b[2,0] * q2, 2 * b[1,0] * q4 - 4 * b[2,0] * q3, 2 * b[1,0] * q3])
            Jb = np.array([[Jb1], [Jb2], [Jb3]]).squeeze()

            fgb = np.concatenate([fg, fb])
            Jgb = np.concatenate([Jg, Jb])

            Df = np.dot(Jgb.transpose(), fgb)

            q_Temp = q - mu * Df / np.linalg.norm(Df)

            q_result = np.array([[q_Temp[0,0]],[q_Temp[1,0]],[q_Temp[2,0]],[q_Temp[3,0]]])

            q_result = q_result / np.linalg.norm(q_result)

            q1 = q_result[0, 0]
            q2 = q_result[1, 0]
            q3 = q_result[2, 0]
            q4 = q_result[3, 0]

            q = np.array([[q1], [q2], [q3], [q4]])

            i = i + 1

        return q

    def update(self, w, a, m, dt):

        """
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
            m - ndarray (dim: 3,1): Magnetometer [-]
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if isinstance(w, list):
            w = np.array(w)

        if len(w.shape) == 1:
            w = w[:, np.newaxis]

        if isinstance(m, list):
            m = np.array(m)

        if len(m.shape) == 1:
            m = m[:, np.newaxis]

        if a.shape[0] != 3 or w.shape[0] != 3 or m.shape[0] != 3:
            print("a, w or m does not have three components")

        self.dt = dt

        a = a / np.linalg.norm(a)
        m = m / np.linalg.norm(m)

        # Predicted states matrix (X_p = A.X + B.U + W)
        const = self.dt / 2
        F1 = np.array([1, -const * w[0,0], -const * w[1,0], -const * w[2,0]])
        F2 = np.array([const * w[0,0], 1, const * w[2,0], -const * w[1,0]])
        F3 = np.array([const * w[1,0], -const * w[2,0], 1, const * w[0,0]])
        F4 = np.array([const * w[2,0], const * w[1,0], -const * w[0,0], 1])
        F = np.array([[F1], [F2], [F3], [F4]]).squeeze()

        self.X_p = np.dot(F, self.X)

        # Predicted covariance matrix (P_p = A.P.A' + Q)
        self.P_p = np.dot(np.dot(F, self.P), F.transpose()) + self.Q

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        K = np.dot(np.dot(self.P_p, self.H.transpose()),
                   np.linalg.inv(np.dot(np.dot(self.H, self.P_p), self.H.transpose()) + self.R))

        # Current observations matrix (Y = C.Y + Z)
        # dq = 0.5 * self.quaternion_product(a=self.X, b=np.array([[0], [w[0,0]], [w[1,0]], [w[2,0]]]))
        # dqnorm = np.linalg.norm(dq)
        # mu = 1 * dqnorm * self.dt

        mu0 = 0.01
        beta = 0.1
        omega = np.linalg.norm(w)
        mu = mu0 + beta * omega * self.dt

        self.z = self.gradient_descent(a, m, self.z, mu)
        self.z = self.z / np.linalg.norm(self.z)

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.dot(K, self.z - np.dot(self.H, self.X_p))
        self.X = self.normalize_quaternion(self.X)

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = np.dot(self.I - np.dot(K, self.H), self.P_p)

        # Information to keep
        self.q = self.X[0:4]
        self.eulers_rad = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(q=self.q, to_degree=False)
        self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

class Gauss_Newton_Kalman_Filter():

    def __init__(self):

        super().__init__()

    def initial(self, dt):

        """
        Parameters:
            dt - Sampling time [sec]
        """

        # Initial values
        self.dt = dt
        self.X = np.array([[1], [0], [0], [0]])
        self.P = np.identity(4) * 2

        # the process covariance matrix
        # var = np.array([(0.5647 / 180 * np.pi) ** 2, (0.5647 / 180 * np.pi) ** 2, (0.5647 / 180 * np.pi) ** 2])
        # Q1 = [var[0] + var[1] + var[2], -var[0] + var[1] - var[2], -var[0] - var[1] + var[2], var[0] - var[2] - var[2]]
        # Q2 = [-var[0] + var[1] - var[2], var[0] + var[1] + var[2], var[0] - var[1] - var[2], -var[0] - var[2] + var[2]]
        # Q3 = [-var[0] - var[1] + var[2], var[0] - var[1] - var[2], var[0] + var[1] + var[2], -var[0] + var[2] - var[2]]
        # Q4 = [var[0] - var[1] - var[2], -var[0] - var[1] + var[2], -var[0] + var[1] - var[2], var[0] + var[2] + var[2]]
        # self.Q = np.array([[Q1], [Q2], [Q3], [Q4]])

        self.Q = np.identity(4) * 0.01

        # the transformation matrix (obsevation　→　state)
        self.H = np.identity(4)

        # the identity matrix
        self.I = np.identity(4)

        # the observations covariance matrix
        self.R = np.identity(4) * 0.01

        # Observation vector
        self.z = np.array([[1], [0], [0], [0]])

        # Information to keep
        self.q = self.X[0:4]
        self.eulers_rad_initial = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(q=self.q, to_degree=False)
        self.eulers_deg_initial = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad_initial)

    def normalize_quaternion(self, q):
        mag = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
        return q / mag

    def quaternion_product(self, a, b):
        """
        外積計算
        :param a: ndarray (4,1)
        :param b: ndarray (4,1)
        :return: ndarray (4,1)
        """

        a1 = a[0, 0]
        a2 = a[1, 0]
        a3 = a[2, 0]
        a4 = a[3, 0]

        b1 = b[0,0]
        b2 = b[1,0]
        b3 = b[2,0]
        b4 = b[3,0]

        p1 = a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4
        p2 = a1 * b2 + a2 * b1 + a3 * b4 - a4 * b3
        p3 = a1 * b3 - a2 * b4 + a3 * b1 + a4 * b2
        p4 = a1 * b4 + a2 * b3 - a3 * b2 + a4 * b1

        P = np.array([[p1], [p2], [p3], [p4]])

        return P

    def compute_jacobian(self, a,b,c,d,Ax,Ay,Az,Mx,My,Mz):

        J11 = 2 * a * Ax + 2 * b * Ay + 2 * c * Az
        J12 = -2 * b * Ax + 2 * a * Ay + 2 * d * Az
        J13 = -2 * c * Ax - 2 * d * Ay + 2 * a * Az
        J14 = 2 * d * Ax - 2 * c * Ay + 2 * b * Az

        J21 = 2 * b * Ax - 2 * a * Ay - 2 * d * Az
        J22 = 2 * a * Ax + 2 * b * Ay + 2 * c * Az
        J23 = 2 * d * Ax - 2 * c * Ay + 2 * b * Az
        J24 = 2 * c * Ax + 2 * d * Ay - 2 * a * Az

        J31 = 2 * c * Ax + 2 * d * Ay - 2 * a * Az
        J32 = -2 * d * Ax + 2 * c * Ay - 2 * b * Az
        J33 = 2 * a * Ax + 2 * b * Ay + 2 * c * Az
        J34 = -2 * b * Ax + 2 * a * Ay + 2 * d * Az

        J41 = 2 * a * Mx + 2 * b * My + 2 * c * Mz
        J42 = -2 * b * Mx + 2 * a * My + 2 * Mz * d
        J43 = -2 * c * Mx - 2 * d * My + 2 * a * Mz
        J44 = 2 * d * Mx - 2 * c * My + 2 * b * Mz

        J51 = 2 * b * Mx - 2 * a * My - 2 * d * Mz
        J52 = 2 * a * Mx + 2 * b * My + 2 * c * Mz
        J53 = 2 * d * Mx - 2 * c * My + 2 * b * Mz
        J54 = 2 * c * Mx + 2 * d * My - 2 * a * Mz

        J61 = 2 * c * Mx + 2 * d * My - 2 * a * Mz
        J62 = -2 * d * Mx + 2 * c * My - 2 * b * Mz
        J63 = 2 * a * Mx + 2 * b * My + 2 * c * Mz
        J64 = -2 * b * Mx + 2 * a * My + 2 * d * Mz

        J = -np.array([[J11, J12, J13, J14],[J21,J22,J23,J24],[J31,J32,J33,J34],[J41,J42,J43,J44],[J51,J52,J53,J54],[J61,J62,J63,J64]])

        return J

    def compute_matrix(self,a,b,c,d):

        R11 = d ** 2 + a ** 2 - b ** 2 - c ** 2
        R12 = 2 * (a * b - c * d)
        R13 = 2 * (a * c + b * d)
        R21 = 2 * (a * b + c * d)
        R22 = d ** 2 + b ** 2 - a ** 2 - c ** 2
        R23 = 2 * (b * c - a * d)
        R31 = 2 * (a * c - b * d)
        R32 = 2 * (b * c + a * d)
        R33 = d ** 2 + c ** 2 - b ** 2 - a ** 2

        R = np.array([[R11, R12, R13],[R21,R22,R23],[R31,R32,R33]])

        zero = np.zeros(shape=(3, 3))

        tmp1 = np.concatenate([R, zero], axis=1)
        tmp2 = np.concatenate([zero, R], axis=1)

        M = np.concatenate([tmp1, tmp2], axis=0)

        return M

    def gauss_newton(self, q, accel, mag):

        a = q[1,0]
        b = q[2,0]
        c = q[3,0]
        d = q[0,0]

        i = 1
        n_k = np.array([[a], [b], [c], [d]])

        n = np.array([[0],[0],[0],[0]])

        while i <= 3 :
            mag = mag / np.linalg.norm(mag)
            q_coniug = np.array([[q[0,0]], [-q[1,0]], [-q[2,0]], [-q[3,0]]])     #共役複素数
            hTemp = self.quaternion_product(q, np.array([[0], [mag[0,0]], [mag[1,0]], [mag[2,0]]]))
            h = self.quaternion_product(hTemp, q_coniug)

            #b = np.array([[math.sqrt(h[1,0] ** 2 + h[2,0] ** 2)], [0], [h[3,0]]])
            bmag = np.array([[0], [math.sqrt(h[1,0] ** 2 + h[2,0] ** 2)], [h[3,0]]])
            bmag = bmag / np.linalg.norm(bmag)

            j_nk = self.compute_jacobian(a, b, c, d, accel[0,0], accel[1,0], accel[2,0], mag[0,0], mag[1,0], mag[2,0])

            M = self.compute_matrix(a, b, c, d)

            y_e = np.concatenate([np.array([[0], [0], [1]]), bmag], axis=0)

            y_b = np.concatenate([accel, mag], axis=0)

            temp1 = np.linalg.inv(np.dot(j_nk.transpose(), j_nk))
            temp2 = np.dot(temp1, j_nk.transpose())
            temp3 = np.dot(M, y_b)
            n = n_k - np.dot(temp2, (y_e - temp3))
            n = n / np.linalg.norm(n)

            a = n[0, 0]
            b = n[1, 0]
            c = n[2, 0]
            d = n[3, 0]
            n_k = n
            q = np.array([[d], [a], [b], [c]])

            i = i + 1

        n = np.array([[n[3,0]], [n[0,0]], [n[1,0]], [n[2,0]]])

        return n

    def update(self, w, a, m, dt):

        """
        Parameters:
            a - ndarray (dim: 3,1): Acceleration [m/sec^2]
            w - ndarray (dim: 3,1): Angular velocity [deg/sec]
            m - ndarray (dim: 3,1): Magnetometer [-]
        """

        if isinstance(a, list):
            a = np.array(a)

        if len(a.shape) == 1:
            a = a[:, np.newaxis]

        if isinstance(w, list):
            w = np.array(w)

        if len(w.shape) == 1:
            w = w[:, np.newaxis]

        if isinstance(m, list):
            m = np.array(m)

        if len(m.shape) == 1:
            m = m[:, np.newaxis]

        if a.shape[0] != 3 or w.shape[0] != 3 or m.shape[0] != 3:
            print("a, w or m does not have three components")

        self.dt = dt

        a = a / np.linalg.norm(a)
        m = m / np.linalg.norm(m)

        # Predicted states matrix (X_p = A.X + B.U + W)
        const = self.dt / 2
        F1 = np.array([1, -const * w[0,0], -const * w[1,0], -const * w[2,0]])
        F2 = np.array([const * w[0,0], 1, const * w[2,0], -const * w[1,0]])
        F3 = np.array([const * w[1,0], -const * w[2,0], 1, const * w[0,0]])
        F4 = np.array([const * w[2,0], const * w[1,0], -const * w[0,0], 1])
        F = np.array([[F1], [F2], [F3], [F4]]).squeeze()

        self.X_p = np.dot(F, self.X)

        # Predicted covariance matrix (P_p = A.P.A' + Q)
        self.P_p = np.dot(np.dot(F, self.P), F.transpose()) + self.Q

        # Kalman gain matrix (K = (P.H') / (H.P.H' + R))
        K = np.dot(np.dot(self.P_p, self.H.transpose()),
                   np.linalg.inv(np.dot(np.dot(self.H, self.P_p), self.H.transpose()) + self.R))

        # Current observations matrix (Y = C.Y + Z)
        self.z = self.gauss_newton(self.z, a, m)
        self.z = self.z / np.linalg.norm(self.z)

        # Current state matrix (X = X_p + K.(Y - H.X))
        self.X = self.X_p + np.dot(K, self.z - np.dot(self.H, self.X_p))
        self.X = self.normalize_quaternion(self.X)

        # Current process covariance matrix (P = (I - K.H).P_p)
        self.P = np.dot(self.I - np.dot(K, self.H), self.P_p)

        # Information to keep
        self.q = self.X[0:4]
        self.eulers_rad = Convertion_angles_rotation_matrix.convert_q_to_euler_angles(q=self.q, to_degree=False)
        self.eulers_deg = Convertion_angles_rotation_matrix.rad2deg(self.eulers_rad)

    def reset_angle(self):
        """
        Reset eulers_rad and eulers_deg to [0, 0, 0]
        """

        self.eulers_rad = np.array([[0], [0], [0]])
        self.eulers_deg = np.array([[0], [0], [0]])

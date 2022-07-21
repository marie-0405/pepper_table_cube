import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from Classes.IMU_Attitude import Convertion_angles_rotation_matrix

class CreateIKdata():
    def __init__(self, high=True):  # If use high quality data, high = True
        self.marker_list = ['FA_A_x', 'FA_A_y', 'FA_A_z', 'FA_BL_x', 'FA_BL_y', 'FA_BL_z', 'FA_BR_x', 'FA_BR_y',
                            'FA_BR_z', 'FA_F_x', 'FA_F_y', 'FA_F_z',  'RIEL_x', 'RIEL_y',
                            'RIEL_z',
                            'RIWR_x', 'RIWR_y', 'RIWR_z',
                            'ROEL_x', 'ROEL_y', 'ROEL_z', 'ROWR_x', 'ROWR_y', 'ROWR_z',
                            'RSH_x', 'RSH_y', 'RSH_z',
                            'UA_A_x', 'UA_A_y', 'UA_A_z', 'UA_BL_x', 'UA_BL_y', 'UA_BL_z', 'UA_BR_x', 'UA_BR_y',
                            'UA_BR_z', 'UA_F_x', 'UA_F_y', 'UA_F_z']

    def calculate_Eular(self, child_coordinate, parent_coordinate, sequence):
        self.euler = {}
        self.c1 = child_coordinate
        self.c2 = parent_coordinate
        self.rotation = np.dot(self.c2[3:12].reshape([3, 3]), np.linalg.inv(self.c1[3:12].reshape([3, 3])))
        self.euler = Convertion_angles_rotation_matrix.convert_R_to_euler_angles(sequence=sequence, R=self.rotation)

    def run(self):
        path_file = "Data\\Motive\\Elbow_Motion_Tracking\\Tetsuya_Abe\\Markers_cut_spline\\Flexion_Extension_opt.npy"
        self.marker_data = np.load(path_file)

        self.marker_dict = {}
        for n in range(len(self.marker_list)):
            self.marker_dict[self.marker_list[n]] = self.marker_data[500:6000, n]

    def make_unit_vector(self, vector_name):

        if vector_name == 'upper_arm':

            markers = self.marker_dict
            # Upper arm vector[O, X, Y, Z]
            # O(Origin): RSH
            # X: Normal vector from the plane made by O, IEL, OEL(Inner and outer elbow), forward positive
            # Y: The vector from the mid of IEL, OEL to Oh,upward positive
            # Z: Perpendicular to two other vectors, right positive

            o = np.array([markers['RSH_x'], markers['RSH_y'], markers['RSH_z']])

            z = np.array([markers['RSH_x'] - (markers['RIEL_x'] + markers['ROEL_x']) / 2,
                          markers['RSH_y'] - (markers['RIEL_y'] + markers['ROEL_y']) / 2,
                          markers['RSH_z'] - (markers['RIEL_z'] + markers['ROEL_z']) / 2])

            y = np.array([np.cross([markers['ROEL_x'] - markers['RSH_x'],
                                    markers['ROEL_y'] - markers['RSH_y'],
                                    markers['ROEL_z'] - markers['RSH_z']],
                                   [markers['RIEL_x'] - markers['RSH_x'],
                                    markers['RIEL_y'] - markers['RSH_y'],
                                    markers['RIEL_z'] - markers['RSH_z']], axis=0)])

            y = y.squeeze().T
            y = y / np.linalg.norm(y, axis=1)[:, np.newaxis]

            z = z.T
            z = z / np.linalg.norm(z, axis=1)[:, np.newaxis]

            x = np.cross(y, z)
            x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

            self.vector = np.empty([x.shape[0], 12])
            self.vector[:, :3] = o.T
            self.vector[:, 3:6] = x
            self.vector[:, 6:9] = y
            self.vector[:, 9:12] = z

        elif vector_name == 'forearm':

            markers = self.marker_dict
            # forearm_vector[O, X, Y, Z]
            # O(Origin): RIWR (Inner wrist)
            # X: Normal vector from the plane made by IWR, OWR, and the mid point of IEL and OEL, forward positive
            # Y: The vector from Of to the mid of IEL, OEL, upward positive
            # Z: Perpendicular to two other vectors, right positive

            o = np.array([markers['RIWR_x'], markers['RIWR_y'], markers['RIWR_z']])

            z = np.array([(markers['RIEL_x'] + markers['ROEL_x']) / 2 - markers['RIWR_x'],
                            (markers['RIEL_y'] + markers['ROEL_y']) / 2 - markers['RIWR_y'],
                            (markers['RIEL_z'] + markers['ROEL_z']) / 2 - markers['RIWR_z']])

            y = np.array([np.cross([markers['ROWR_x'] - (markers['RIEL_x'] + markers['ROEL_x']) / 2,
                                      markers['ROWR_y'] - (markers['RIEL_y'] + markers['ROEL_y']) / 2,
                                      markers['ROWR_z'] - (markers['RIEL_z'] + markers['ROEL_z']) / 2],
                                     [markers['RIWR_x'] - (markers['RIEL_x'] + markers['ROEL_x']) / 2,
                                      markers['RIWR_y'] - (markers['RIEL_y'] + markers['ROEL_y']) / 2,
                                      markers['RIWR_z'] - (markers['RIEL_z'] + markers['ROEL_z']) / 2], axis=0)])

            y = y.squeeze().T
            y = y / np.linalg.norm(y, axis=1)[:, np.newaxis]

            z = z.T
            z = z / np.linalg.norm(z, axis=1)[:, np.newaxis]

            x= np.cross(y, z)
            x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

            # z = np.array([(markers['RIEL_x'] + markers['ROEL_x']) / 2 - markers['RIWR_x'],
            #                 (markers['RIEL_y'] + markers['ROEL_y']) / 2 - markers['RIWR_y'],
            #                 (markers['RIEL_z'] + markers['ROEL_z']) / 2 - markers['RIWR_z']])
            #
            # x = np.array([np.cross([markers['RIWR_x'] - (markers['RIEL_x'] + markers['ROEL_x']) / 2,
            #                         markers['RIWR_y'] - (markers['RIEL_y'] + markers['ROEL_y']) / 2,
            #                         markers['RIWR_z'] - (markers['RIEL_z'] + markers['ROEL_z']) / 2],
            #                        [markers['ROWR_x'] - (markers['RIEL_x'] + markers['ROEL_x']) / 2,
            #                         markers['ROWR_y'] - (markers['RIEL_y'] + markers['ROEL_y']) / 2,
            #                         markers['ROWR_z'] - (markers['RIEL_z'] + markers['ROEL_z']) / 2], axis=0)])
            #
            # x = x.squeeze().T
            # x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
            #
            # z = z.T
            # z = z / np.linalg.norm(z, axis=1)[:, np.newaxis]
            #
            # y = np.cross(x, z)
            # y = y / np.linalg.norm(y, axis=1)[:, np.newaxis]

            self.vector = np.empty([x.shape[0], 12])
            self.vector[:, :3] = o.T
            self.vector[:, 3:6] = x
            self.vector[:, 6:9] = y
            self.vector[:, 9:12] = z

        elif vector_name == 'upper_arm_device':

            markers = self.marker_dict
            # forearm_device_vector[O, X, Y, Z]
            # O(Origin): UA_F
            # X: UA_F → UA_BR
            # Y: Perpendicular to two other vectors
            # Z: Normal vector from the plane(UA_F) made by UA_F, UA_BL, UA_BR

            o = np.array([markers['UA_F_x'], markers['UA_F_y'], markers['UA_F_z']])

            y = np.array([markers['UA_BR_x'] - markers['UA_F_x'], markers['UA_BR_y'] - markers['UA_F_y'],
                          markers['UA_BR_z'] - markers['UA_F_z']])
            x = np.array(
                [np.cross([markers['UA_BL_x'] - markers['UA_F_x'], markers['UA_BL_y'] - markers['UA_F_y'],
                           markers['UA_BL_z'] - markers['UA_F_z']],
                          [markers['UA_BR_x'] - markers['UA_F_x'], markers['UA_BR_y'] - markers['UA_F_y'],
                           markers['UA_BR_z'] - markers['UA_F_z']],
                          axis=0)])

            x = x.squeeze().T
            x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

            y = y.T
            y = y / np.linalg.norm(y, axis=1)[:, np.newaxis]

            z = np.cross(x, y)
            z = z / np.linalg.norm(z, axis=1)[:, np.newaxis]

            self.vector = np.concatenate((o.T, x), axis=1)
            self.vector = np.concatenate((self.vector, y), axis=1)
            self.vector = np.concatenate((self.vector, z), axis=1)

        else:

            markers = self.marker_dict
            # forearm_device_vector[O, X, Y, Z]
            # O(Origin): FA_BL
            # X: FA_BL → FA_BR
            # Y: Perpendicular to two other vectors
            # Z: Normal vector from the plane(FA_BL) made by FA_F, FA_BL, FA_BR

            o = np.array([markers['FA_BL_x'], markers['FA_BL_y'], markers['FA_BL_z']])

            y = np.array([markers['FA_BR_x'] - markers['FA_BL_x'], markers['FA_BR_y'] - markers['FA_BL_y'],
                          markers['FA_BR_z'] - markers['FA_BL_z']])
            x = np.array(
                [np.cross([markers['FA_BR_x'] - markers['FA_BL_x'], markers['FA_BR_y'] - markers['FA_BL_y'],
                           markers['FA_BR_z'] - markers['FA_BL_z']],
                          [markers['FA_F_x'] - markers['FA_BL_x'], markers['FA_F_y'] - markers['FA_BL_y'],
                           markers['FA_F_z'] - markers['FA_BL_z']],
                          axis=0)])

            x = x.squeeze().T
            x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

            y = y.T
            y = y / np.linalg.norm(y, axis=1)[:, np.newaxis]

            z = np.cross(x, y)
            z = z / np.linalg.norm(z, axis=1)[:, np.newaxis]

            self.vector = np.concatenate((o.T, x), axis=1)
            self.vector = np.concatenate((self.vector, y), axis=1)
            self.vector = np.concatenate((self.vector, z), axis=1)

    def calculate_IK(self, parent_coordinate, child_coordinate):

        self.c1 = child_coordinate
        self.c2 = parent_coordinate
        if parent_coordinate == 'base':
            self.c2 = np.eye(3)  # OMC reference

        self.rotation = []
        for l in range(self.c1.shape[0]):
            self.rotation.append(np.dot(self.c2, np.linalg.inv(self.c1[l].reshape(3, 3))))

# UA = CreateIKdata()
# UA.run()
# UA.make_unit_vector(vector_name='upper_arm')
# UA.calculate_IK(parent_coordinate='base', child_coordinate=UA.vector[:, 3:])
#
# FA = CreateIKdata()
# FA.run()
# FA.make_unit_vector(vector_name='forearm')
# FA.calculate_IK(parent_coordinate='base', child_coordinate=FA.vector[:, 3:])
#
# UA_D = CreateIKdata()
# UA_D.run()
# UA_D.make_unit_vector(vector_name='upper_arm_device')
# UA_D.calculate_IK(parent_coordinate='base', child_coordinate=UA_D.vector[:, 3:])
#
# FA_D = CreateIKdata()
# FA_D.run()
# FA_D.make_unit_vector(vector_name='forearm_device')
# FA_D.calculate_IK(parent_coordinate='base', child_coordinate=FA_D.vector[:, 3:])




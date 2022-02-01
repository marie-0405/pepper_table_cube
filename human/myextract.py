import cv2
import mediapipe as mp
import os
import pandas as pd
from datetime import datetime
import numpy as np

FILE_NAME = "BodyLanguage"

# MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Input the video
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture('./resources/{}.mp4'.format(FILE_NAME)) # 0 for web camera input

TimeStamp = [] # Time stamps list in real-time

# Define (x, y) coordinates lists
X_LShoulder = []  # Left Shoulder X
X_RShoulder = []  # Right Shoulder X
X_LElbow = []  # Left Elbow X
X_RElbow = []  # Right Elbow X
X_LWrist = []  # Left Wrist X
X_RWrist = []  # Right Wrist X
X_LHip = []  # Left Hip X
X_RHip = []  # Right Hip X
# X_LKnee = []
# X_RKnee = []
# X_LAnkle = []
# X_RAnkle = []
X_Nose = []  # Nose X

Y_LShoulder = []  # Left Shoulder Y
Y_RShoulder = []  # Right Shoulder Y
Y_LElbow = []  # Left Elbow Y
Y_RElbow = []  # Right Elbow Y
Y_LWrist = []  # Left Wrist Y
Y_RWrist = []  # Right Wrist Y
Y_LHip = []  # Left Hip Y
Y_RHip = []  # Right Hip Y
# Y_LKnee = []
# Y_RKnee = []
# Y_LAnkle = []
# Y_RAnkle = []
Y_Nose = []  # Nose Y

Z_LShoulder = []  # Left Shoulder Z
Z_RShoulder = []  # Right Shoulder Z
Z_LElbow = []  # Left Elbow Z
Z_RElbow = []  # Right Elbow Z
Z_LWrist = []  # Left Wrist Z
Z_RWrist = []  # Right Wrist Z
Z_LHip = []  # Left Hip Z
Z_RHip = []  # Right Hip Z
# Z_LKnee = []
# Z_RKnee = []
# Z_LAnkle = []
# Z_RAnkle = []
Z_Nose = []  # Nose Z

# Loop for each frame
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Get the current time
    time = datetime.now()
    TimeStamp.append(time)

    # Get the pose coordinates of each landmark
    image_height, image_width, _ = image.shape # Normalize the reference frame according to the resolution of the video.

    # Get the left shoulder coordinates
    X11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
    Y11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
    Z11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z

    X_LShoulder.append(X11)
    Y_LShoulder.append(Y11)
    Z_LShoulder.append(Z11)

    # Get the right shoulder coordinates
    X12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    Y12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
    Z12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z

    X_RShoulder.append(X12)
    Y_RShoulder.append(Y12)
    Z_RShoulder.append(Z12)

    # Get the left elbow coordinates
    X13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width
    Y13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height
    Z13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z

    X_LElbow.append(X13)
    Y_LElbow.append(Y13)
    Z_LElbow.append(Z13)

    # Get the right elbow coordinates
    X14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
    Y14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height
    Z14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z

    X_RElbow.append(X14)
    Y_RElbow.append(Y14)
    Z_RElbow.append(Z14)

    # Get the left wrist coordinates
    X15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width
    Y15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height
    Z15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z

    X_LWrist.append(X15)
    Y_LWrist.append(Y15)
    Z_LWrist.append(Z15)

    # Get the right wrist coordinates
    X16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width
    Y16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height
    Z16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z

    X_RWrist.append(X16)
    Y_RWrist.append(Y16)
    Z_RWrist.append(Z16)

    # Get the left hip coordinates
    X23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
    Y23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
    Z23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z

    X_LHip.append(X23)
    Y_LHip.append(Y23)
    Z_LHip.append(Z23)

    # Get the right hip coordinates
    X24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
    Y24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height
    Z24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z

    X_RHip.append(X24)
    Y_RHip.append(Y24)
    Z_RHip.append(Z24)

    # X25 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width
    # Y25 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height
    # Z25 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z
    #
    # X_LKnee.append(X25)
    # Y_LKnee.append(Y25)
    # Z_LKnee.append(Z25)
    #
    # X26 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width
    # Y26 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height
    # Z26 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z
    #
    # X_RKnee.append(X26)
    # Y_RKnee.append(Y26)
    # Z_RKnee.append(Z26)
    #
    # X27 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width
    # Y27 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height
    # Z27 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z
    #
    # X_LAnkle.append(X27)
    # Y_LAnkle.append(Y27)
    # Z_LAnkle.append(Z27)
    #
    # X28 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width
    # Y28 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height
    # Z28 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z
    #
    # X_RAnkle.append(X28)
    # Y_RAnkle.append(Y28)
    # Z_RAnkle.append(Z28)

    # Get the nose coordinates
    X0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
    Y0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
    Z0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z

    X_Nose.append(X0)
    Y_Nose.append(Y0)
    Z_Nose.append(Z0)

# Close MediaPipe Pose
pose.close()
cap.release()

# Joint Angles in 2D
xy = {'LeftShoulder': np.column_stack([X_LShoulder, Y_LShoulder]), 'RightShoulder': np.column_stack([X_RShoulder, Y_RShoulder]),
      'LeftElbow': np.column_stack([X_LElbow, Y_LElbow]), 'RightElbow': np.column_stack([X_RElbow, Y_RElbow]),
      'LeftWrist': np.column_stack([X_LWrist, Y_LWrist]), 'RightWrist': np.column_stack([X_RWrist, Y_RWrist]),
      'LeftHip': np.column_stack([X_LHip, Y_LHip]), 'RightHip': np.column_stack([X_RHip, Y_LHip])}

# Total no. of seconds elapsed in execution
delta = TimeStamp[-1] - TimeStamp[0]
sec = delta.total_seconds()
num = sec / (len(TimeStamp))

timestamps = []

for i in range(len(TimeStamp)):
    s = (num * i) + num
    timestamps.append(s)

Frame = []
for i in range(len(timestamps)):
    Frame.append(int(i))


# Arms landmarks
landmark1 = 'LeftShoulder'
landmark2 = 'LeftHip'
landmark3 = 'LeftElbow'
landmark4 = 'LeftWrist'
landmark5 = 'RightShoulder'
landmark6 = 'RightHip'
landmark7 = 'RightElbow'
landmark8 = 'RightWrist'


# Robot/Human Angles
LShoulderPitch = []
RShoulderPitch = []
LElbowYaw = []
RElbowYaw = []
RShoulderRoll = []
LShoulderRoll = []
RElbowRoll = []
LElbowRoll = []
HeadYaw = []
HeadPitch = []
HipRoll = []
HipPitch = []


# Calculate each body segment vector in 2D
LS_LE = xy[landmark3] - xy[landmark1]  # Left Shoulder to Left Elbow
LE_LS = xy[landmark1] - xy[landmark3]  # Left Elbow to Left Shoulder
LW_LE = xy[landmark3] - xy[landmark4]  # Left Wrist to Left Elbow
LS_LH = xy[landmark2] - xy[landmark1]  # Left Shoulder to Left Hip

RS_RE = xy[landmark7] - xy[landmark5]  # Right Shoulder to Right Elbow
RE_RS = xy[landmark5] - xy[landmark7]  # Right Elbow to Right Shoulder
RW_RE = xy[landmark7] - xy[landmark8]  # Right Wrist to Right Elbow
RS_RH = xy[landmark6] - xy[landmark5]  # Right Shoulder to Right Hip

RS_LS = xy[landmark1] - xy[landmark5]  # Right Shoulder to Left Shoulder

torso = Y_RHip[0] - Y_RShoulder[0]  # Torso original height

# Calculate the hip roll angles
for i in range(len(X_LShoulder)):
    # Current horizontal distance between the 2 shoulders over the distance between them
    adj = (X_LShoulder[i] - X_RShoulder[i]) / np.linalg.norm(RS_LS[i, :])
    if adj >= 1:          # Keeping the ratio less than or equal to 1
        adj = 1
    phi = np.arccos(adj)  # Arc cos to get the angle
    if phi >= 0.5149:     # Maximum right hip roll is 29.5°.
        phi = 0.5149
    if Y_LShoulder[i] < Y_RWrist[i]:     # If right shoulder is above the left shoulder then the direction of hip roll is reversed.
        phi = phi * -1
    if phi <= -0.5149:    # Maximum left hip roll is -29.5°.
        phi = -0.5149
    HipRoll.append(phi)

# Calculate the hip pitch angles
for i in range(len(Y_LShoulder)):
    adj = (Y_RHip[i] - Y_LShoulder[i]) / torso  # Current height of torso over the original height
    if adj >= 1:                   # Keeping the ratio less than or equal to 1
        adj = 1
    phi = np.arccos(adj)           # Hip pitch angle for leaning forward is negative
    if phi >= 1.0385:              # Maximum hip pitch angle is 59.5°.
        phi = 1.0385
    HipPitch.append(phi)

# Calculate the head yaw angles
d = np.linalg.norm(RS_LS[0, :]) / 2  # Half of initial distance between right and left shoulder

for i in range(len(X_Nose)):
    if (X_Nose[i] - X_RShoulder[i]) / d >= 0.9 and (X_Nose[i] - X_RShoulder[i]) / d <= 1.1:  # Estimating the angle to be 0° if the nose
        hy = 0.0                                                   # X coordinate doesn't exceed 10% from each side

    elif (X_Nose[i] - X_RShoulder[i]) / d < 0.9:                                # Angle of looking to the right based on how much
        hy = ((d - (X_Nose[i] - X_RShoulder[i])) / d) * -(np.pi / 2)            # the nose is approaching the right shoulder.
        if hy <= -np.pi / 2:                                       # Maximum head yaw angle to the right is -90°.
            hy = -np.pi / 2

    elif (X_Nose[i] - X_RShoulder[i]) / d > 1.1:                                # Angle of looking to the right based on how much
        hy = (((X_Nose[i] - X_RShoulder[i]) - d) / d) * (np.pi / 2)             # the nose is approaching the left shoulder.
        if hy >= np.pi / 2:                                        # Maximum head yaw angle to the left is 90°.
            hy = np.pi / 2
    HeadYaw.append(hy)

# Calculate the head pitch angles
h = Y_RShoulder[0] - Y_Nose[0]
for i in range(len(Y_Nose)):
    if (Y_RShoulder[i] - Y_Nose[i]) / h >= 0.95 and (Y_RShoulder[i] - Y_Nose[i]) / h <= 1.0:
        hp = 0.0
    elif (Y_RShoulder[i] - Y_Nose[i]) / h < 0.95:
        hp = ((h - (Y_RShoulder[i] - Y_Nose[i])) / h) * 0.6371
        if hp >= 0.6371:
            hp = 0.6371
    elif (Y_RShoulder[i] - Y_Nose[i]) / h > 1.0:
        hp = (((Y_RShoulder[i] - Y_Nose[i]) - h) / h) * -0.7068 * 2
        if hp <= -0.7068:
            hp = -0.7068
    HeadPitch.append(hp)

# 3D coordinates
xyz = {'LeftShoulder': np.column_stack([X_LShoulder, Y_LShoulder, Z_LShoulder]),
       'RightShoulder': np.column_stack([X_RShoulder, Y_RShoulder, Z_RShoulder]),
       'LeftElbow': np.column_stack([X_LElbow, Y_LElbow, Z_LElbow]), 'RightElbow': np.column_stack([X_RElbow, Y_RElbow, Z_RElbow]),
       'LeftWrist': np.column_stack([X_LWrist, Y_LWrist, Z_LWrist]), 'RightWrist': np.column_stack([X_RWrist, Y_RWrist, Z_RWrist])}

# 3D vectors
LS_LE_3D = xyz[landmark3] - xyz[landmark1]
RS_RE_3D = xyz[landmark7] - xyz[landmark5]

LE_LS_3D = xyz[landmark1] - xyz[landmark3]
LW_LE_3D = xyz[landmark3] - xyz[landmark4]

RE_RS_3D = xyz[landmark5] - xyz[landmark7]
RW_RE_3D = xyz[landmark7] - xyz[landmark8]

UpperArmLeft = xyz[landmark3] - xyz[landmark1]
UpperArmRight = xyz[landmark7] - xyz[landmark5]

ZeroXLeft = xyz[landmark3] - xyz[landmark1]
ZeroXRight = xyz[landmark7] - xyz[landmark5]

ZeroXLeft[:, 0] = 0
ZeroXRight[:, 0] = 0

UpperArmLeft[:, 1] = 0
UpperArmRight[:, 1] = 0

# Original Lengths of arm segments
l1_left = np.linalg.norm(LE_LS[0, :])  # Upper left arm
l2_left = np.linalg.norm(LW_LE[0, :])  # Lower left arm

l1_right = np.linalg.norm(RE_RS[0, :])  # Upper right arm
l2_right = np.linalg.norm(RW_RE[0, :])  # Lower right arm


# Calculate the left shoulder roll angles
for i in range(LS_LE_3D.shape[0]):
    temp1 = (np.dot(LS_LE_3D[i, :], ZeroXLeft[i, :])) / (np.linalg.norm(LS_LE_3D[i, :]) * np.linalg.norm(ZeroXLeft[i, :]))
    temp = np.arccos(temp1)
    if temp >= 1.56:
        temp = 1.56
    if temp <= np.arccos((np.dot(LS_LE_3D[0, :], ZeroXLeft[0, :])) / (np.linalg.norm(LS_LE_3D[0, :]) * np.linalg.norm(ZeroXLeft[0, :]))):
        temp = 0.0
    LShoulderRoll.append(temp)

# Calculate the right shoulder roll angles
for i in range(RS_RE_3D.shape[0]):
    temp1 = (np.dot(RS_RE_3D[i, :], ZeroXRight[i, :])) / (np.linalg.norm(RS_RE_3D[i, :]) * np.linalg.norm(ZeroXRight[i, :]))
    temp = np.arccos(temp1)
    if temp >= 1.56:
        temp = -1.56
    else:
        temp = temp * (-1)
    if temp > -np.arccos((np.dot(RS_RE_3D[0, :], ZeroXRight[0, :])) / (np.linalg.norm(RS_RE_3D[0, :]) * np.linalg.norm(ZeroXRight[0, :]))):
        temp = 0.0
    RShoulderRoll.append(temp)

# Calculate the left elbow roll angles
for i in range(LE_LS_3D.shape[0]):
    temp1 = (np.dot(LE_LS_3D[i, :], LW_LE_3D[i, :])) / (np.linalg.norm(LE_LS_3D[i, :]) * np.linalg.norm(LW_LE_3D[i, :]))
    temp = np.arccos(temp1)
    if temp >= 1.56:
        temp = -1.56
    else:
        temp = temp * -1
    LElbowRoll.append(temp)

# Calculate the right elbow roll angles
for i in range(RE_RS_3D.shape[0]):
    temp1 = (np.dot(RE_RS_3D[i, :], RW_RE_3D[i, :])) / (np.linalg.norm(RE_RS_3D[i, :]) * np.linalg.norm(RW_RE_3D[i, :]))
    temp = np.arccos(temp1)
    if temp >= 1.56:
        temp = 1.56
    RElbowRoll.append(temp)

# Calculate the left shoulder pitch & left elbow yaw angles
for i in range(LE_LS_3D.shape[0]):
    temp1 = (np.dot(UpperArmLeft[i, :], LS_LE_3D[i, :])) / (np.linalg.norm(UpperArmLeft[i, :]) * np.linalg.norm(LS_LE_3D[i, :]))
    temp = np.arccos(temp1)
    if temp >= np.pi / 2:
        temp = np.pi / 2
    if Y_LShoulder[i] > Y_LElbow[i]:
        temp = temp * -1
    LShoulderPitch.append(temp)

    if LShoulderRoll[i] <= 0.4:
        ley = -np.pi / 2
    elif Y_LElbow[i] - Y_LWrist[i] > 0.2 * l2_left:
        ley = -np.pi / 2
    elif Y_LElbow[i] - Y_LWrist[i] < 0 and -(Y_LElbow[i] - Y_LWrist[i]) > 0.2 * l2_left and LShoulderRoll[i] > 0.7:
        ley = np.pi / 2
    else:
        ley = 0.0
    LElbowYaw.append(ley)

# Calculate the right shoulder pitch & right elbow yaw angles
for i in range(RE_RS_3D.shape[0]):
    temp1 = (np.dot(UpperArmRight[i, :], RS_RE_3D[i, :])) / (np.linalg.norm(UpperArmRight[i, :]) * np.linalg.norm(RS_RE_3D[i, :]))
    temp = np.arccos(temp1)
    if temp >= np.pi / 2:
        temp = np.pi / 2
    if Y_RShoulder[i] > Y_RElbow[i]:
        temp = temp * -1
    RShoulderPitch.append(temp)

    if RShoulderRoll[i] >= -0.4:
        rey = np.pi / 2
    elif Y_RElbow[i] - Y_RWrist[i] > 0.2 * l2_right:
        rey = np.pi / 2
    elif Y_RElbow[i] - Y_RWrist[i] < 0 and -(Y_RElbow[i] - Y_RWrist[i]) > 0.2 * l2_right and RShoulderRoll[i] < -0.7:
        rey = -np.pi / 2
    else:
        rey = 0.0
    RElbowYaw.append(rey)

# Total no. of seconds elapsed in execution
delta = TimeStamp[-1] - TimeStamp[0]
sec = delta.total_seconds()
num = sec / (len(TimeStamp))

timestamps = []

T = []
LSP = []
RSP = []
REY = []
LEY = []
LSR = []
LER = []
RSR = []
RER = []
HY = []
HP = []
HPP = []
HPR = []

# Human joint angles
AngleHuman = [LShoulderPitch, RShoulderPitch, RElbowYaw, LElbowYaw, LShoulderRoll, LElbowRoll,
              RShoulderRoll, RElbowRoll, HeadYaw, HeadPitch, HipRoll, HipPitch]

# Robot joint angles
AngleRobot = [LSP, RSP, REY, LEY, LSR, LER, RSR, RER, HY, HP, HPR, HPP]

# Shorten the timestamps
# 7 can be changed. Minimum 10 is preferred
for i in range(len(TimeStamp)):
    s = (num * i) + num
    timestamps.append(s)
n = int((len(timestamps) - (len(timestamps) % 12)) / 12)
if len(timestamps) % 12 != 0:
    N = int(n + 1)
else:
    N = n

for j in range(1, N):
    t = timestamps[(j + 1) * 12 - 12]
    T.append(t)
if len(timestamps) % 12 != 0:
    T.append(T[-1] + T[0])

# Shorten the joint angles
def ShortenData(AngleHuman, AngleRobot):
    for i in range(len(AngleRobot)):
        for j in range(1, N):
            theta = AngleHuman[i][(j + 1) * 12 - 12]
            AngleRobot[i].append(theta)
        if len(timestamps) % 12 != 0:
            AngleRobot[i].append(AngleHuman[i][-1])

ShortenData(AngleHuman, AngleRobot)

def SafePosition(T, AngleRobot):
    T.append(T[-1] + 2)  # 2 seconds to reach the safe position

    for i in range(4, len(AngleRobot)):
        AngleRobot[i].append(0.0)

    for i in range(3):
        AngleRobot[i].append(np.pi / 2)

    AngleRobot[3].append(-np.pi / 2)


SafePosition(T, AngleRobot)

# Export data to input to Pepper
angles = {'TimeStamp': T, 'LShoulderRoll': LSR, 'LElbowRoll': LER,
          'RShoulderRoll': RSR, 'RElbowRoll': RER, 'HeadYaw': HY,
          'HeadPitch': HP, 'LShoulderPitch': LSP, 'RShoulderPitch': RSP,
          'LElbowYaw': LEY, 'RElbowYaw': REY, 'HipRoll': HPR, 'HipPitch': HPP}

# 3D coordinates
xyz = {'Frame': Frame, 'Time': timestamps,
       'X1': X_LShoulder, 'Y1': Y_LShoulder, 'Z1': Z_LShoulder,
       'X2': X_RShoulder, 'Y2': Y_RShoulder, 'Z2': Z_RShoulder,
       'X3': X_LElbow, 'Y3': Y_LElbow, 'Z3': Z_LElbow,
       'X4': X_RElbow,'Y4':  Y_RElbow,'Z4': Z_RElbow,
       'X5': X_LWrist, 'Y5': Y_LWrist, 'Z5': Z_LWrist,
       'X6': X_RWrist, 'Y6': Y_RWrist, 'Z6': Z_RWrist,
       'X7': X_LHip, 'Y7': Y_LHip, 'Z7': Z_LHip,
       'X8': X_RHip, 'Y8': Y_RHip, 'Z8': Z_RHip,
       'X9': X_Nose, 'Y9': Y_Nose, 'Z9': Z_Nose}

cwd = os.getcwd()
XYZ = pd.DataFrame.from_dict(xyz)
XYZ.to_csv(cwd + '\\resources\\{}.csv'.format(FILE_NAME), index=False)




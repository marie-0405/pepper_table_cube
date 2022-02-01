# 環境構築
mediapipeはpip install でインストールできる。
```
pip install mediapipe
```
mdeiapipeや他のopencv-pythonなどをインストールする場合、Python27だとうまくいかないので、extract.pyや3DMotion.pyはPython3の環境で実行すること！


# extract
## MediaPipe Poseの定義
```python
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
```

## poseの定義
```python
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
```

## whileの開始
```python
X11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
    Y11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
    Z11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z
```
mediapipeによって、x, y, zの全ての座標を取得できる。Pepperなので、下半身の一部は取得しない。

# Close MediaPipe Pose
```
pose.close()
cap.release()
```
poseはcloseする必要がある。

## 3D coordinates
```python
xyz = {'Frame': Frame, 'Time': timestamps,
       'X1': X_LShoulder, 'Y1': Y_LShoulder, 'Z1': Z_LShoulder,
       'X2': X_RShoulder, 'Y2': Y_RShoulder, 'Z2': Z_RShoulder,
}
XYZ.to_csv(cwd + '\\..\\resources\\obama.csv', index=False)
```
3Dの座標を辞書型で定義して、csvファイルで出力。だから、csvファイルにあるのは、3次元座標のみ。

# 3次元座標から、角度を求めたい！
* Hishamのコードを参考にする
  
# poseの引数を色々変えて、どうなるのか調べてみたい！
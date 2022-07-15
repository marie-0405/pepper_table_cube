import numpy as np

class Body_R():

    @staticmethod
    def R_trunk(SJN, SXS, CV7, TV7):

        Ot = SJN
        Yt = (Ot + CV7) * 0.5 - (SXS + TV7) * 0.5
        Yt = Yt / np.linalg.norm(Yt)  # 正規化
        Zt = np.cross(Ot - CV7, Yt)
        Zt = Zt / np.linalg.norm(Zt)  # 正規化
        Xt = np.cross(Yt, Zt)
        Rt = np.hstack([Xt.reshape(-1, 1), Yt.reshape(-1, 1), Zt.reshape(-1, 1)])

        return Ot, Xt, Yt, Zt, Rt

    @staticmethod
    def R_humerus(CAJ, HLE, HME, Yt, way):

        Oh1 = CAJ - Yt * 0.07  # "GH"marker
        Yh1 = Oh1 - (HLE + HME) * 0.5
        Yh1 = Yh1 / np.linalg.norm(Yh1)  # 正規化

        if way == "right":

            Xh1 = np.cross(Yh1, HLE - HME)

        elif way == "left":

            Xh1 = np.cross(Yh1, HME - HLE)

        Xh1 = Xh1 / np.linalg.norm(Xh1)  # 正規化
        Zh1 = np.cross(Xh1, Yh1)
        Rh1 = np.hstack([Xh1.reshape(-1, 1), Yh1.reshape(-1, 1), Zh1.reshape(-1, 1)])

        return Oh1, Xh1, Yh1, Zh1, Rh1

    @staticmethod
    def R_humerus2(CAJ, HLE, HME, Yt, Yf):

        Oh2 = CAJ - Yt * 0.07  # "GH"marker
        Yh2 = Oh2 - (HLE + HME) * 0.5
        Yh2 = Yh2 / np.linalg.norm(Yh2)  # 正規化
        Zh2 = np.cross(Yh2, Yf)
        Zh2 = Zh2 / np.linalg.norm(Zh2)  # 正規化
        Xh2 = np.cross(Yh2, Zh2)
        Rh2 = np.hstack([Xh2.reshape(-1, 1), Yh2.reshape(-1, 1), Zh2.reshape(-1, 1)])

        return Oh2, Xh2, Yh2, Zh2, Rh2

    @staticmethod
    def R_forearm(USP, HLE, HME, RSP, way):

        Of = USP
        Yf = (HLE + HME) * 0.5 - Of  # forearmのy軸(尺骨茎状突起からELとEMの中点)
        Yf = Yf / np.linalg.norm(Yf)  # 正規化

        if way == "right":

            Xf = np.cross(RSP - Of, Yf)  # forearmのx軸

        elif way == "left":

            Xf = np.cross(Of - RSP, Yf)  # forearmのx軸

        Xf = Xf / np.linalg.norm(Xf)  # 正規化
        Zf = np.cross(Xf, Yf)
        Rf = np.hstack([Xf.reshape(-1, 1), Yf.reshape(-1, 1), Zf.reshape(-1, 1)])  # Xfなどは一次元のndarrayのため、.Tが効かない

        return Of, Xf, Yf, Zf, Rf

    @staticmethod
    def R_pelvis(RIAS, LIAS, RIPS, LIPS):

        Op = (RIAS + LIAS) * 0.5
        Zp = RIAS - LIAS
        Zp = Zp / np.linalg.norm(Zp)  # 正規化
        Yp = np.cross(Zp, Op - (RIPS + LIPS) * 0.5)
        Yp = Yp / np.linalg.norm(Yp)  # 正規化
        Xp = np.cross(Yp, Zp)
        Rp = np.hstack([Xp.reshape(-1, 1), Yp.reshape(-1, 1), Zp.reshape(-1, 1)])

        return Op, Xp, Yp, Zp, Rp

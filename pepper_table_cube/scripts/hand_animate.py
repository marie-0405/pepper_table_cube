#!/usr/bin/env python

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')

line, = ax.plot([], [], 'o-', linewidth=2) # このlineに次々と座標を代入して描画

def animate(i):
    thisx = [L * np.cos(PHI[i]) * np.cos(THETA[i]),L * np.cos(PHI[i+1]) * np.cos(THETA[i+1])]
    thisy = [L * np.cos(PHI[i]) * np.sin(THETA[i]), L * np.cos(PHI[i+1]) * np.sin(THETA[i+1])]
    thisz = [L * np.sin(PHI[i]), L * np.sin(PHI[i+1])]
    # 1ステップ前の位置から現在位置へのlineとする。
    line.set_data(thisx, thisy)
    line.set_3d_properties(thisz)
    return line,

ani = FuncAnimation(fig, animate, frames=np.arange(0, len(t)), interval=25, blit=True)

# ax.set_xlim(-5, np.cos(L * np.cos(THETA[stepCnt-1])))
ax.set_xlim(-3.5, 3.5)
# ax.set_ylim(L * np.sin(L * THETA[stepCnt-1]),10)
ax.set_ylim(-3.5, 3.5)
ax.set_zlim(-3.5, 3.5)

# グラフを作成
fig2 = plt.figure(figsize = (8, 8))
plt.plot(t, THETA[:60])
plt.xlabel('Time [s]')
plt.ylabel('θ[rad]')
fig3 = plt.figure(figsize = (8, 8)) 
plt.plot(t, PHI[:60])
plt.xlabel('Time [s]')
plt.ylabel('φ[rad]')
fig2.savefig('/content/drive/My Drive/PSE/θのグラフ')
fig3.savefig('/content/drive/My Drive/PSE/φのグラフ')

# 軸ラベルを設定
ax.set_xlabel("x", size = 14)
ax.set_ylabel("y", size = 14)
ax.set_zlabel("z", size = 14)
ax.grid()
plt.show()

ani.save('/content/drive/My Drive/PSE/Spherical_pendulum(L = 0.1m).gif', writer='pillow', fps=15)
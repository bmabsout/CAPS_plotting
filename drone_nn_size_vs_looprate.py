import matplotlib.pyplot as plt

x = [128,64,32,16,8,4]
x = range(6)

plt.plot(x, [400,730,850,925,963,967], linewidth=3, marker='o', markersize=10)
plt.grid()
plt.xticks(x,["128x128", "64x64", "32x32", "16x16", "8x8", "4x4"])
plt.xlabel("Feedforward neural network size")
plt.ylabel("Loop rate (Hz)")
plt.ylim([350,1000])
plt.show()
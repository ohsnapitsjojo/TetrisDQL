import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt( "log_file.txt", delimiter="," )

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.plot(data[:,0], 'r-')
ax1.set_title('Time')

ax2 = fig.add_subplot(222)
ax2.plot(data[:,1], 'k-')
ax2.set_title('Reward')

ax3 = fig.add_subplot(223)
ax3.plot(data[:,2], 'b-')
ax3.set_title('Score')

ax4 = fig.add_subplot(224)
ax4.plot(data[:,3], 'g-')
ax4.set_title('Lines')
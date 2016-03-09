#####
#
# Working with multiple figure windows and subplots
#
#####

import matplotlib.pyplot as plt
import numpy as np

# set up time axis and sine wave functions
t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)


plt.figure(1)
plt.gcf().suptitle('Bold Title', fontsize=14, fontweight='bold')
plt.gcf().canvas.set_window_title('Test')

plt.subplot(321) # 3 row, 2 column, in plot location 1
plt.plot(t, s1)

plt.subplot(322)
plt.plot(t, 2*s1)

plt.subplot(324)
plt.plot(t, 4*s1)

#another plot
plt.figure(2)
plt.gcf().canvas.set_window_title('Test 2')
plt.plot(t, s2)

# now switch back to figure 1 and make some changes
plt.figure(1)
plt.gca().set_xticklabels(['h', 'g', 'a'])  # gca = get current axis
plt.subplot(321)
plt.plot(t, s2)

plt.show()

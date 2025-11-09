import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import MultipleLocator

plt.rc('font',family='Times New Roman', size=4)

head = np.loadtxt('head.txt', dtype=np.float32)
tail = np.loadtxt('tail.txt', dtype=np.float32)

#设置子图个数和整个图片的大小
dpi=450
fig, axes = plt.subplots(nrows=1, ncols=2,figsize = (900/dpi, 450/dpi), dpi=dpi, sharey=False)

#设置colorbar的范围
vmin = 0
vmax = 1.7
norm = colors.Normalize(vmin=vmin, vmax=vmax)

x_major_locator=MultipleLocator(3)
y_major_locator=MultipleLocator(3)

x_minor_locator=MultipleLocator(1)
y_minor_locator=MultipleLocator(1)

sub_fig_0 = axes[0].imshow(head, norm = norm, origin='lower',cmap = plt.cm.jet)
axes[0].xaxis.set_major_locator(x_major_locator)
axes[0].yaxis.set_major_locator(y_major_locator)
axes[0].xaxis.set_minor_locator(x_minor_locator)
axes[0].yaxis.set_minor_locator(y_minor_locator)

axes[0].tick_params(which='minor', width=0.15, length=0.5)
axes[0].tick_params(which='major', width=0.3, length=1, pad=0.5)
axes[0].spines['top'].set_linewidth('0.05')
axes[0].spines['bottom'].set_linewidth('0.05')
axes[0].spines['left'].set_linewidth('0.05')
axes[0].spines['right'].set_linewidth('0.05')


sub_fig_1 = axes[1].imshow(tail, norm = norm, origin='lower',cmap = plt.cm.jet)
axes[1].xaxis.set_major_locator(x_major_locator)
axes[1].yaxis.set_major_locator(y_major_locator)
axes[1].xaxis.set_minor_locator(x_minor_locator)
axes[1].yaxis.set_minor_locator(y_minor_locator)

axes[1].tick_params(which='major', width=0.3, length=1, pad=0.5)
axes[1].tick_params(which='minor', width=0.15, length=0.5)
axes[1].spines['top'].set_linewidth('0.05')
axes[1].spines['bottom'].set_linewidth('0.05')
axes[1].spines['left'].set_linewidth('0.05')
axes[1].spines['right'].set_linewidth('0.05')

# fig.subplots_adjust(right=0.9)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)

cbar_ax = fig.add_axes([axes[1].get_position().x1+0.01, axes[1].get_position().y0, 0.02, axes[1].get_position().height])

fc = fig.colorbar(sub_fig_1, cax=cbar_ax)
fc.outline.set_linewidth(0.3)
cax = fc.ax
cax.tick_params(width=0.3, length=1, pad=0.5)
# fc.set_ticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8])
plt.savefig('./descriptors.pdf', bbox_inches='tight',pad_inches=0.01)
plt.show()

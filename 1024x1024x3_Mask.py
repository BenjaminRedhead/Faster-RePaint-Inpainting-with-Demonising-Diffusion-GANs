import torch
import torch.nn as nn
import numpy as np
import re
from google.colab import files
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math



n = np.random.randint(1,15)     #8 # Number of possibly sharp edges
r = np.random.randint(1,15)/15  #.7 # magnitude of the perturbation from the unit circle, 
# should be between 0 and 1
N = n*3+1 # number of points in the Path
# There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

angles = np.linspace(0,2*np.pi,N)
codes = np.full(N,Path.CURVE4)
codes[0] = Path.MOVETO

verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]
verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
path = Path(verts, codes)

fig = plt.figure(figsize=(1024/72,1024/72))
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax.add_patch(patch)



ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
ax.axis('off') # removes the axis to leave only the shape

#plt.show()


def canvas2rgb_array(canvas):
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    return buf.reshape(scale * nrows, scale * ncols, 3)


plt_array = canvas2rgb_array(fig.canvas)
print(plt_array.shape)
#print(plt_array)



th, im_th = cv2.threshold(plt_array, 200, 255, cv2.THRESH_BINARY_INV)
im_floodfill = im_th.copy()

h, w = im_th.shape[:2]

mask = np.zeros((h+2,w+2),np.uint8)
cv2.floodFill(im_floodfill, mask, (0,0),(255,255,255))

cv2_imshow(im_floodfill)

#print(im_floodfill)




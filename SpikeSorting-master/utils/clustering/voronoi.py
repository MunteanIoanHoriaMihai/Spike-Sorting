from PIL import Image
import random
import math
 
def generate_voronoi_diagram(width, height, num_cells):
	image = Image.new("RGB", (width, height))
	putpixel = image.putpixel
	imgx, imgy = image.size
	nx = []
	ny = []
	nr = []
	ng = []
	nb = []
	for i in range(num_cells):
		nx.append(random.randrange(imgx))
		ny.append(random.randrange(imgy))
		nr.append(random.randrange(256))
		ng.append(random.randrange(256))
		nb.append(random.randrange(256))
	for y in range(imgy):
		for x in range(imgx):
			dmin = math.hypot(imgx-1, imgy-1)
			j = -1
			for i in range(num_cells):
				d = math.hypot(nx[i]-x, ny[i]-y)
				if d < dmin:
					dmin = d
					j = i
			putpixel((x, y), (nr[j], ng[j], nb[j]))
	image.save("VoronoiDiagram.png", "PNG")
	image.show()
 
generate_voronoi_diagram(500, 500, 25)

import numpy as np
import pandas as pd
import math
import random

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d

import utils.clustering.trim.approximation_scheme as fs

# Importing the dataset
data = pd.read_csv('../../datasets/data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

#print("Shape:")
#print(data.shape)
#print("\n")

c1 = np.array([f1]).T
c2 = np.array([f2]).T
c3 = np.array([f3]).T


X = np.append(c1, c3, axis=1)
X = fs.approximationScheme(X)
initialX = X

vor = Voronoi(X)
voronoi_plot_2d(vor)
plt.show()

# points = np.array([[-5, -2], [-2, 3] , [1, -2], [3, -2], [4, -1], [5, 6]])
# vor = Voronoi(points)
# voronoi_plot_2d(vor)
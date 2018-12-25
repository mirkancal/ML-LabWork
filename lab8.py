#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

#%% Find and append pixels to the 

im = Image.open("black white flower.jpg")
im = im.convert(mode="1")
im = im.resize((320,256))

width, height = im.size

Xc = np.array([])
Yc = np.array([])

for i in range(0, width):
    for j in range(0, height):
        if (im.getpixel((i,j)) != 0):
            Xc = np.append(Xc, i)
            Yc = np.append(Yc, j)
            
            
plt.scatter(Xc, Yc)

X = np.vstack((Xc, Yc)).T

kmeans = KMeans(n_clusters=4).fit(X)

pred = kmeans.predict(X)

c1_x = np.array([])
c1_y = np.array([])

c2_x = np.array([])
c2_y = np.array([])

c3_x = np.array([])
c3_y = np.array([])

c4_x = np.array([])
c4_y = np.array([])


count = 0

for p in pred:
    if p == 0:
        c1_x = np.append(c1_x, Xc[count])
        c1_y = np.append(c1_y, Yc[count])
    if p == 1:
        c2_x = np.append(c2_x, Xc[count])
        c2_y = np.append(c2_y, Yc[count])
    if p == 2:    
        c3_x = np.append(c3_x, Xc[count])
        c3_y = np.append(c3_y, Yc[count])
    if p == 3:
        c4_x = np.append(c4_x, Xc[count])
        c4_y = np.append(c4_y, Yc[count])
    count += 1
    

plt.figure()
plt.scatter(c1_x, c1_y, color="blue")
plt.scatter(c2_x, c2_y, color="red")
plt.scatter(c3_x, c3_y, color="green")
plt.scatter(c4_x, c4_y, color="olive")

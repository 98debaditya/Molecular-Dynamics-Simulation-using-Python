import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as dis
np.seterr(divide='ignore', invalid='ignore')

A = 1 #LJ constent
B = 1 #LJ constant
M = 1 #Mass of molucule

#creating 1/d as array
def dist(r):
    d = dis(r,r)
    shape_d = d.shape
    d = 1/d
    d = d.reshape(500*500)
    d[d == np.inf] = 0
    return d.reshape(shape_d)

#Creating force array
def force(r):
    fx = r[:,0]*(12*A*dist(r)**14 - 6*B*dist(r)**8)
    fx = fx.sum(axis=1)
    fy = r[:,1]*(12*A*dist(r)**14 - 6*B*dist(r)**8)
    fy = fy.sum(axis=1)
    fz = r[:,2]*(12*A*dist(r)**14 - 6*B*dist(r)**8)
    fz = fz.sum(axis=1)
    f = np.array([fx,fy,fz])
    return np.transpose(f)

#Updating r & v using Verlet algorithom
def verlet(r,v):
    f1 = force(r)
    r1 = r + v*0.001 + 0.5*(force(r)/M)*0.001**2
    f2 = force(r)
    v1 = v + 0.5*((f1 + f2)/M)*0.001
    return r1,v1

#Collition with wall and molucules
def col_wall(r):
    r1 = np.copy(r)
    shape_r = r1.shape
    r1 = r1.reshape(500*3)
    r1[r1 > 135] = -1
    r1[r1 < -135] = -1
    r1[r1 != -1] = 1
    r1 = r1.reshape(shape_r)
    return np.multiply(r1,v)

r = np.random.uniform(-135,135,size=[500,3]) #generating position array
v = np.random.uniform(-500,500,size=[500,3]) #generating velocity array

#starting iteration
R = [r]
V = [v]
for i in range(0,1000):
    print('Generating array',round(i*100/999,1),"%", end='\r')
    r,v = verlet(r,v)
    v = col_wall(r)
    R = R + [r]
    V = V + [v]
print()

R = np.array(R)
V = np.array(V)

#generating frames
for i in range (len(R)):
    print('Generating frames',round(i*100/(len(R)-1),1),"%", end='\r')
    ax = plt.axes(projection ="3d")
    x,y,z = R[i].T
    ax.axes.set_xlim3d(left=-135, right=135) 
    ax.axes.set_ylim3d(bottom=-135, top=135) 
    ax.axes.set_zlim3d(bottom=-135, top=135)
    ax.scatter3D(x,y,z, color = "green")
    plt.savefig('/home/deb/database/%s.png'%f"{i:04}")
    plt.clf()
print()

#generating video
import cv2
import os
path = '/home/deb/database/'
video_out = '/home/deb/video1_out.mp4'

frm = os.listdir(path)
frm=sorted(frm)
vtype = cv2.VideoWriter_fourcc(*'mp4v')
size = cv2.imread(path+frm[0]).shape
size = list(size)
del size[2]
size.reverse()

video = cv2.VideoWriter(video_out,vtype,30,size)

for i in range(len(frm)):
    print('Generating video',round(i*100/(len(frm)-1),1),"%", end='\r')
    video.write(cv2.imread(path + frm[i]))
print()
video.release()

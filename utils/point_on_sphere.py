#!/usr/bin/env python3

# Module to distribute points on a sphere approximately equi-distant from each
# other. Written in C by Paul Bourke, July 1996.
# Translated in Python by Emmanuel Koumandakis, 2018.
# TODO: use Eigen, Numpy
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import pyplot as plt
import random, math

class XYZ:
    def __init__(self,x=0,y=0,z=0):
        self.x = x
        self.y = y
        self.z = z

def normalize(XYZ, r):
    '''Args:
        XYZ: a 3D vector
        r: scale factor

        returns XYZ*(r/norm2(XYZ))'''
    l = r/(math.sqrt(XYZ.x**2+XYZ.y**2+XYZ.z**2))
    XYZ.x = XYZ.x*l
    XYZ.y = XYZ.y*l
    XYZ.z = XYZ.z*l
    return XYZ

def dist(v1, v2):
    x = v1.x - v2.x
    y = v1.y - v2.y
    z = v1.z - v2.z
    return (math.sqrt(x**2+y**2+z**2))

def distr_points(n, r, countmax):
    '''Args:
        n: number of points to distribute
        r: radius of sphere
        countmax: number of iterations
    '''
    # Check input
    if (n < 2):
        n = 3
    if (r < 0.001):
        r = 0.001
    if (countmax < 100):
        countmax = 100

    # Initialize points
    p = []
    for i in range(n):
        p.append(XYZ())
    p1 = XYZ(1,1,1)
    p2 = XYZ()

    # Create the initial random cloud
    for i in range(n):
        x = random.randrange(-500, 500,1)
        y = random.randrange(-500, 500,1)
        z = random.randrange(-500, 500,1)
        p[i] = normalize(XYZ(x,y,z), r)

    for cnt in range(countmax):
        # Find the closest two points
        minp1 = 0 # index
        minp2 = 1 # index
        mind = dist(p[minp1],p[minp2])
        maxd = mind
        for i in range(n-1):
            for j in range(i+1, n):
                d = dist(p[i],p[j])
                if (d < mind):
                    mind = d
                    minp1 = i
                    minp2 = j
                if (d > maxd):
                    maxd = d

        if (cnt - countmax == -1):
            print('rand dist: ', dist(p[minp1],p[minp2]))

        # Move the two points apart by 5%
        # TODO: vary this for refinement
        p1 = p[minp1]
        p2 = p[minp2]
        p[minp2].x = p1.x + 1.05*(p2.x - p1.x)
        p[minp2].y = p1.y + 1.05*(p2.y - p1.y)
        p[minp2].z = p1.z + 1.05*(p2.z - p1.z)
        p[minp1].x = p1.x - 0.05*(p2.x - p1.x)
        p[minp1].y = p1.y - 0.05*(p2.y - p1.y)
        p[minp1].z = p1.z - 0.05*(p2.z - p1.z)
        p[minp1] = normalize(p[minp1], r)
        p[minp2] = normalize(p[minp2], r)
        print('Progress: ', 100*(cnt/countmax),'%')

    return p # list of point coords

def write_to_file(p, filename):
    f = open(filename, "w")
    txt = ""
    for pt in p:
        txt = txt+str(pt.x)+','+str(pt.y)+','+str(pt.z)+'\n'
    f.write(txt)
    f.close()

def main():
    # Distribute points on a sphere and plot them
    n = 300
    r = 1
    c = 10000
    filename = 'coords_pts_'+str(n)+'_its_'+str(c)+'.txt'
    p = distr_points(n,r,c)

    write_to_file(p, filename)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(p)):
        ax.scatter(p[i].x,p[i].y,p[i].z, c='b')

    plt.show()



if __name__=="__main__":
    main()


import numpy as np
import math 

def Distance(trials, file):
    #opens the point cloud and converts it to floats.
    f=open(file, "r")
    if f.mode == 'r':
        cloud = f.read()

    cloud = cloud.split(' ')

    for i in range(16):
        del cloud[0]

    #indervidually remove these phrases because joined to first desired element in an unknown way
    cloud[0] = cloud[0].replace('blue', '')
    cloud[0] = cloud[0].replace('end_header', '')
    cloud[0] = cloud[0].replace('\n', '')

    del cloud[-1]

    float_cloud = []

    for j in cloud:
        j = j.replace('\n','')
        j = float(j)
        float_cloud.append(j)

    #finds the minimum and maximum x and y points to find centre co-ordinate
    xpos = []
    ypos = []
    zpos = []

    for n1 in (np.arange(6,(len(float_cloud) + 1),6) - 6):
        xpos.append(float_cloud[n1])

    for n2 in (np.arange(6,(len(float_cloud) + 1),6) - 5):
        ypos.append(float_cloud[n2])

    for n3 in (np.arange(6,(len(float_cloud) + 1),6) - 4):
        zpos.append(float_cloud[n3])

    #For the number of trials finding the shortest path from the centre and then excluding this value from the next search, finds the trial number of closest points
    close_pos = []
    for t in range(trials):
        shortest_dist = 1.0
        positions = []
        for n in range(len(xpos)):
            dist = math.sqrt((xpos[n] * xpos[n]) + (ypos[n] * ypos[n]))
            if dist < shortest_dist:
                shortest_dist = dist
                positions.append(n)
        close_pos.append(positions[-1])
        del xpos[positions[-1]]
        del ypos[positions[-1]]

    total_dist = 0
    for listno in close_pos:
        total_dist += zpos[listno]
        print(zpos[listno])

    

    average_dist = total_dist / trials
    average_dist = average_dist * 1.023
    print('Distance to object is:',average_dist)
    

Distance(20, "output/point_clouds/L_rect_l__point_cloud.ply")




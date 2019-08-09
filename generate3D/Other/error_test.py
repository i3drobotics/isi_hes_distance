import cv2
import numpy as np
import time
import glob
from matplotlib import pyplot as plt
import argparse
import signal
import math
import os
import pptk

#number is number of pictures taken with an interval in secs between pictures
#trials is number of points in the point array chaosen to find the distance to
number = 1
interval = 0
trials = 15

dists = []

Scal = cv2.FileStorage("cal/stereo_calibration.xml", cv2.FILE_STORAGE_READ)
Rot_matrix = Scal.getNode("R").mat()
Tran_dist = Scal.getNode("T").mat()
Scal.release()

#loads in the calibration files and finds the camera matrix and the distortion coefficients
Rcal = cv2.FileStorage("cal/Right_calibration.xml", cv2.FILE_STORAGE_READ)
Rmatrix = Rcal.getNode("cameraMatrix").mat()
Rdist = Rcal.getNode("distCoeffs").mat()
Rcal.release()

Lcal = cv2.FileStorage("cal/left_calibration.xml", cv2.FILE_STORAGE_READ)
Lmatrix = Lcal.getNode("cameraMatrix").mat()
Ldist = Lcal.getNode("distCoeffs").mat()
Lcal.release()

#finds camera and launches window
cam = cv2.VideoCapture(0)
cv2.namedWindow("stereoCamera; press space for capture")

count = 0
imagesL = np.array([[]])
imagesR = np.array([[]])

while True:
    ret, frame = cam.read()

    #combines left and right images together
    frame_left = frame.copy()
    frame_left = cv2.flip(frame_left[:, :, 1],0)
    frame_right = frame.copy()
    frame_right = cv2.flip(frame_right[:, :, 2],0)
    
    frame_joint = np.concatenate((frame_left,frame_right), axis=1)

    cv2.imshow("stereoCamera; press space for picture", frame_joint)

    if not ret: 
        break
    k = cv2.waitKey(1)

    count = 1
    
    if k%256 == 32:
        count += 1
        
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        
        data = dists
        data = np.array(data)
        data = (data - 90.6247) / 0.7982

        print(data)

        total = 0
        for i in data:
            total += i

        average = total / len(data)
        print(average)

        print()
        varience = 0

        for i in data:
            x = i - average
            x = x * x
            varience += x / len(data)

        stdev = varience ** 0.5
        print(stdev)
        break

    if count > 0 and count <= number:
        #pressing space takes picture and applies calibration
        #print('taking picture and applying calibration...')
        imgL = frame_left
        imgR = frame_right
        
        Lheight, Lwidth = imgL.shape[:2]
        Rheight, Rwidth = imgR.shape[:2]

        (leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                Lmatrix, Ldist,
                Rmatrix, Rdist,
                (Lwidth,Lheight), Rot_matrix, Tran_dist,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY, -1)

        #undistort
        Lmapx,Lmapy = cv2.initUndistortRectifyMap(Lmatrix,Ldist,leftRectification,leftProjection,(Lwidth,Lheight),5)
        Ldst = cv2.remap(imgL,Lmapx,Lmapy,cv2.INTER_LINEAR)

        Rmapx,Rmapy = cv2.initUndistortRectifyMap(Rmatrix,Rdist,rightRectification,rightProjection,(Rwidth,Rheight),5)
        Rdst = cv2.remap(imgR,Rmapx,Rmapy,cv2.INTER_LINEAR)

        #print('saving rectified images in: input...')
        strcnt = str(count)
        
        globals()['image'+str(count)+'_l_'] = Ldst
        globals()['image'+str(count)+'_r_'] = Rdst

        time.sleep(interval)
        count += 1
        
    if count > number:
        str_number = str(number)
        str_interval = str(interval)
        #print(str_number+' pictures taken, at '+str_interval+' intervals')
        #print('saved variables in form: image()_r_ or image()_l_')
        count = 0

        for n in range(number):
            n += 1

            ply_header = '''
            '''

            CV_MATCHER_BM = 1
            CV_MATCHER_SGBM = 0

            #define matching parameters
            #SETTINGS FOR PHOBOS NUCLEAR  
            algorithm = CV_MATCHER_BM
            window_size = 30
            block_size = 15
            min_disp = 20
            num_disp = 16*6
            uniqness_ratio = 15
            speckle_window_size = 500
            speckle_range = 5

            #define command line arguments
            parser = argparse.ArgumentParser()
            parser.add_argument("-c","--calibration_folder",
                                help="folder location of calibration file",
                                type=str,default='cal')
            parser.add_argument("-i","--input_folder",
                                help="folder location of left and right images",
                                type=str,default='input')
            parser.add_argument("-p","--output_folder_point_clouds",
                                help="folder location to output point clouds",
                                type=str,default='output/point_clouds')
            parser.add_argument("-m","--output_folder_disparity",
                                help="folder location to output disparity maps",
                                type=str,default='output/disparity')
            parser.add_argument("-l","--left_wildcard",
                                help="wildcard for reading images from left camera in folder",
                                type=str,default='*_l_*.png')
            parser.add_argument("-r","--right_wildcard",
                                help="wildcard for reading images from right camera in folder",
                                type=str,default='*_r_*.png')
            parser.add_argument("-x","--pose_wildcard",
                                help="wildcard for reading camera pose in folder",
                                type=str,default='*.txt')
            parser.add_argument("-v","--visualise3D",
                                help="visualise point cloud",
                                type=bool,default=False)
            parser.add_argument("-d","--visualise_disparity",
                                help="visualise disparity map",
                                type=bool,default=True)
            parser.add_argument("-t","--pose_transformation",
                                help="enable transformation of generated point clouds by pose (pose file for each image pair containing [x,y,z,w,x,y,z])",
                                type=bool,default=False) 
            args = parser.parse_args()

            def write_ply(fn, verts, colors):
                verts = verts.reshape(-1, 3)
                colors = colors.reshape(-1, 3)

                verts = np.hstack([verts, colors])
                
                with open(fn, 'wb') as f:
                    f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
                    np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

            def visualise_points(visualiser,verts,colors):
                verts = verts.reshape(-1, 3)
                colors = colors.reshape(-1, 3)

                visualiser.clear()
                visualiser.load(verts)
                visualiser.attributes(colors / 255.)

            def t_q_to_matrix(translation,quaternion):
                """Return homogeneous rotation matrix from quaternion.

                >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
                >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
                True
                >>> M = quaternion_matrix([1, 0, 0, 0])
                >>> numpy.allclose(M, numpy.identity(4))
                True
                >>> M = quaternion_matrix([0, 1, 0, 0])
                >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
                True

                """
                q = np.array(quaternion, dtype=np.float64, copy=True)
                n = np.dot(q, q)
                _EPS = np.finfo(float).eps * 4.0
                if n < _EPS:
                    return np.identity(4)
                q *= math.sqrt(2.0 / n)
                q = np.outer(q, q)
                return np.array([
                    [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], translation[0]],
                    [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], translation[1]],
                    [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], translation[2]],
                    [                0.0,                 0.0,                 0.0, 1.0]])

            def transform_points(points,transform_matrix):
                dMax,dimensions = points.shape
                transformed_points = np.zeros_like(points)
                for d in range(0,dMax):
                    point = points[d]
                    point2 = np.append(point,1)
                    transformed_point = np.matmul(transform_matrix,point2)
                    transformed_point = np.array([transformed_point[0],transformed_point[1],transformed_point[2]])
                    transformed_points[d] = transformed_point
                return(transformed_points)
                        
            def main():

                if (args.visualise3D):
                    init_xyz = pptk.rand(10, 3)
                    visualiser = pptk.viewer(init_xyz)
                
                try:
                    #load calibration file
                    cal_xml = args.calibration_folder + '/stereo_calibration.xml'
                    fs = cv2.FileStorage(cal_xml,flags=cv2.FILE_STORAGE_READ)
                    Q = fs.getNode("Q").mat()
                    #print("Q\n", Q)
                    fs.release()

                    #load camera images
                    left_fns = [1,1]
                    right_fns = [1,1]
                    pose_fns = [1,1]

                    #check the same number of left and right images exist
                    if (not(len(left_fns) == len(right_fns))):
                        raise ValueError("Should have the same number of left and right images")
                        
                    if (args.pose_transformation):
                        if (not(len(left_fns) == len(pose_fns))):
                            raise ValueError("Should have the same number of image as pose files")

                    i = 0
                    while i < len(left_fns):
                        left_fn = left_fns[i]
                        right_fn = right_fns[i]
                        if (args.pose_transformation):
                            pose_fn = pose_fns[i]

                        left_fn_basename = 'image'+str(n)+'_l_'

                        #print("reading images...")
                        #read left and right image from file list
                        imgL = globals()['image'+str(n)+'_l_']
                        imgR = globals()['image'+str(n)+'_r_']
                

                        # Convert source image to unsigned 8 bit integer Numpy array
                        arrL = np.uint8(imgL)
                        arrR = np.uint8(imgR)

                        #print(arrL.shape)
                        #print(arrR.shape)

                        #print("stereo matching...")
                        
                        #generate disparity using stereo matching algorithms
                        if algorithm == CV_MATCHER_BM:
                            stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
                            stereo.setMinDisparity(min_disp)
                            stereo.setSpeckleWindowSize(speckle_window_size)
                            stereo.setSpeckleRange(speckle_range)
                            stereo.setUniquenessRatio(uniqness_ratio)
                        elif algorithm == CV_MATCHER_SGBM:
                            stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                    numDisparities = num_disp,
                                    blockSize = block_size,
                                    P1 = 8*3*window_size**2,
                                    P2 = 32*3*window_size**2,
                                    disp12MaxDiff = 1,
                                    uniquenessRatio = uniqness_ratio,
                                    speckleWindowSize = speckle_window_size,
                                    speckleRange = speckle_range
                                    )
                        
                        disp = stereo.compute(arrL, arrR).astype(np.float32) / 16.0

                        #print("generating 3D...")
                        #reproject disparity to 3D
                        points = cv2.reprojectImageTo3D(disp, Q)

                        #print("saving disparity maps...")

                        disp = (disp-min_disp)/num_disp
                        #cv2.imwrite(args.output_folder_disparity + "/{}_disparity_map.png".format(left_fn_basename),disp)

                        #if (args.visualise_disparity):
                            #dispay disparity to window
                            #plt.imshow(disp)
                            #plt.show(block=False)
                            #plt.pause(0.1)

                        #normalise disparity
                        imask = disp > disp.min()
                        disp_thresh = np.zeros_like(disp, np.uint8)
                        disp_thresh[imask] = disp[imask]

                        disp_norm = np.zeros_like(disp, np.uint8)
                        cv2.normalize(disp, disp_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        #cv2.imwrite(args.output_folder_disparity + "/{}_disparity_image.png".format(left_fn_basename),disp_norm)

                        #format colour image from left camera for mapping to point cloud
                        h, w = arrL.shape[:2]
                        colors = cv2.cvtColor(arrL, cv2.COLOR_BGR2RGB)
                        mask = disp > disp.min()
                        out_points = points[mask]
                        out_colors = colors[mask]

                        globals()['points_'+str(n)] = out_points
                        
                        out_points_fn = args.output_folder_point_clouds + '/{}_point_cloud.ply'.format(left_fn_basename)
                        out_points_transformed_fn = args.output_folder_point_clouds + '/{}_point_cloud_transformed.ply'.format(left_fn_basename)

                        if (args.pose_transformation):
                            #print("transforming point cloud...")
                            
                            #extract pose from pose file
                            pose_file = open(pose_fn,'r')
                            line = pose_file.readline().rstrip()
                            pose = line.split(',')
                            if (not (len(pose) == 7)):
                                error_msg = "Invalid number of values in pose data\nShould be in format [x,y,z,w,x,y,z]" 
                                raise ValueError(error_msg)
                            pose_np = np.array([float(pose[0]),float(pose[1]),float(pose[2]),\
                                                float(pose[3]),float(pose[4]),float(pose[5]),float(pose[6])])

                            #print("transformation:")
                            #print(pose_np)
                            #get tranlation and quaternion
                            pose_t = np.array([float(pose[0]),float(pose[1]),float(pose[2])])
                            pose_q = np.array([float(pose[4]),float(pose[5]),float(pose[6]),float(pose[3])])
                            pose_matrix = t_q_to_matrix(pose_t,pose_q)
                            #print("transformation matrix:")
                            #print(pose_matrix)

                            transformed_points = transform_points(out_points,pose_matrix)

                        i += 1

                    if (args.visualise3D):
                        visualiser.close()
                    if (args.visualise_disparity):
                        plt.close()
                    
                except KeyboardInterrupt:
                    if (args.visualise3D):
                        visualiser.close()
                    if (args.visualise_disparity):
                        plt.close()
                    raise KeyboardInterrupt()
                    
            if __name__ == '__main__':
                main()
        #print('Saved point clouds...')

        average_distance = 0
        for n in range(number):
            
            n += 1
            globals()['xpos_'+str(n)] = []
            globals()['ypos_'+str(n)] = []
            globals()['zpos_'+str(n)] = []
            globals()['alt_xpos_'+str(n)] = []
            globals()['alt_ypos_'+str(n)] = []
            
            for pos in range(len(globals()['points_'+str(n)])):
                globals()['alt_xpos_'+str(n)].append(globals()['points_'+str(n)][pos][0])
                globals()['alt_ypos_'+str(n)].append(globals()['points_'+str(n)][pos][1])
                globals()['zpos_'+str(n)].append(globals()['points_'+str(n)][pos][2])

            #Distance needs to be adjusted from the distance of the left camera to the distance from the probe.
            x_difference = -0.032
            y_difference = -0.073
            
            for x in globals()['alt_xpos_'+str(n)]:
                x = x + x_difference
                globals()['xpos_'+str(n)].append(x)

            for y in globals()['alt_ypos_'+str(n)]:
                y = y + y_difference
                globals()['ypos_'+str(n)].append(y)

                
            close_pos = []    
            for t in range(trials):
                shortest_dist = 10.0
                positions = []
                for N in range(len(globals()['xpos_'+str(n)])):
                    dist = math.sqrt((globals()['xpos_'+str(n)][N] * globals()['xpos_'+str(n)][N]) + (globals()['ypos_'+str(n)][N] * globals()['ypos_'+str(n)][N]))
                    if dist < shortest_dist:
                        shortest_dist = dist
                        positions.append(N)
                close_pos.append(positions[-1])
                del globals()['xpos_'+str(n)][positions[-1]]
                del globals()['ypos_'+str(n)][positions[-1]]

            globals()['totdist_'+str(n)] = 0
            for listno in close_pos:
                globals()['totdist_'+str(n)] += globals()['zpos_'+str(n)][listno]
            average_distance += (globals()['totdist_'+str(n)] / trials)
            #print((globals()['totdist_'+str(n)] / trials))
            
        #cameras are set back 25mm from probe
        """
       [1282.0, 1280.5, 1276.5, 1277.6, 1276.0, 1275.2, 1274.1, 1269.3, 1267.4, 1266.8, 1266.5, 1266.2, 1266.2, 1262.4, 1260.5, 1262.7, 1262.7, 1262.1, 1265.4, 1262.6, 1261.0, 1259.9, 1258.2, 1258.5, 1256.2, 1258.5, 1260.2, 1252.9, 1257.0, 1256.7, 1258.1, 1257.4, 1255.1, 1256.2, 1253.2, 1259.8, 1258.7, 1257.9, 1259.3, 1254.8, 1255.2, 1255.5, 1253.5, 1257.3, 1264.3, 1251.7, 1255.4, 1201.7, 1253.4, 1258.0, 1263.6, 1253.7, 1259.5, 1259.0, 1205.2, 1258.8, 1258.9, 1259.4, 1261.8, 1254.8, 1255.6, 1265.9, 1260.9, 1256.2, 1261.8, 1258.4, 1260.5, 1261.3, 1257.6, 1258.4, 1259.9, 1261.7, 1263.0, 1262.9, 1258.2, 1264.1, 1264.1, 1263.8, 1261.2, 1259.1, 1265.2, 1263.2, 1260.8, 1265.1, 1259.1]
        1282.0
        1201.7
        80.29999999999995
        1260.0117647058821
        58.31176470588207
        21.988235294117885

        10.695809094216791
        """

        average_distance = (average_distance / number) * 1000
        average_distance = average_distance
        average_distance = round((average_distance),1)
        
        dists.append(average_distance)
        



        
                    
            

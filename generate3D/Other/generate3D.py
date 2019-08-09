import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import argparse
import signal
import math
import os
import pptk

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
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
        
#SETTINGS FOR deimos
'''algorithm = CV_MATCHER_SGBM
window_size = 3
block_size = 15
min_disp = 0
num_disp = 64-min_disp
uniqness_ratio = 3
speckle_window_size = 500
speckle_range = 5
'''

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
        print("Q\n", Q)
        fs.release()

        #load camera images
        left_fns = glob.glob(args.input_folder + '/' + args.left_wildcard)
        right_fns = glob.glob(args.input_folder + '/' + args.right_wildcard)
        pose_fns = glob.glob(args.input_folder + '/' + args.pose_wildcard)

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
                print(pose_fn)

            left_fn_basename = os.path.splitext(os.path.basename(left_fn))[0]
            print(left_fn_basename)

            print("reading images...")
            #read left and right image from file list
            imgL = cv2.imread(left_fn,cv2.IMREAD_GRAYSCALE)
            imgR = cv2.imread(right_fn,cv2.IMREAD_GRAYSCALE)
            print(imgL)
            print(len(imgL[0]))
    

            # Convert source image to unsigned 8 bit integer Numpy array
            arrL = np.uint8(imgL)
            arrR = np.uint8(imgR)

            #print(arrL.shape)
            #print(arrR.shape)

            print("stereo matching...")
            
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

            print("generating 3D...")
            #reproject disparity to 3D
            points = cv2.reprojectImageTo3D(disp, Q)

            print("saving disparity maps...")

            disp = (disp-min_disp)/num_disp
            cv2.imwrite(args.output_folder_disparity + "/{}_disparity_map.png".format(left_fn_basename),disp)

            if (args.visualise_disparity):
                #dispay disparity to window
                plt.imshow(disp)
                plt.show(block=False)
                plt.pause(0.1)

            #normalise disparity
            imask = disp > disp.min()
            disp_thresh = np.zeros_like(disp, np.uint8)
            disp_thresh[imask] = disp[imask]

            disp_norm = np.zeros_like(disp, np.uint8)
            cv2.normalize(disp, disp_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            cv2.imwrite(args.output_folder_disparity + "/{}_disparity_image.png".format(left_fn_basename),disp_norm)

            #format colour image from left camera for mapping to point cloud
            h, w = arrL.shape[:2]
            colors = cv2.cvtColor(arrL, cv2.COLOR_BGR2RGB)
            mask = disp > disp.min()
            out_points = points[mask]
            out_colors = colors[mask]
            
            out_points_fn = args.output_folder_point_clouds + '/{}_point_cloud.ply'.format(left_fn_basename)
            out_points_transformed_fn = args.output_folder_point_clouds + '/{}_point_cloud_transformed.ply'.format(left_fn_basename)

            if (args.pose_transformation):
                print("transforming point cloud...")
                
                #extract pose from pose file
                pose_file = open(pose_fn,'r')
                line = pose_file.readline().rstrip()
                pose = line.split(',')
                if (not (len(pose) == 7)):
                    error_msg = "Invalid number of values in pose data\nShould be in format [x,y,z,w,x,y,z]" 
                    raise ValueError(error_msg)
                pose_np = np.array([float(pose[0]),float(pose[1]),float(pose[2]),\
                                    float(pose[3]),float(pose[4]),float(pose[5]),float(pose[6])])

                print("transformation:")
                print(pose_np)
                #get tranlation and quaternion
                pose_t = np.array([float(pose[0]),float(pose[1]),float(pose[2])])
                pose_q = np.array([float(pose[4]),float(pose[5]),float(pose[6]),float(pose[3])])
                pose_matrix = t_q_to_matrix(pose_t,pose_q)
                #print("transformation matrix:")
                #print(pose_matrix)

                transformed_points = transform_points(out_points,pose_matrix)
                

            if (args.visualise3D):
                #visualise point cloud
                visualise_points(visualiser,out_points,out_colors)
                
            if (args.pose_transformation):
                print("saving point clouds...")
                write_ply(out_points_transformed_fn, transformed_points, out_colors)
            else:
                print("saving point cloud...")
            write_ply(out_points_fn, out_points, out_colors)

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


# This script estimates the camera pose from the apriltags in a sequence of images.

import argparse
import apriltag
from glob import glob
import cv2
import numpy as np
from os.path import join
import pandas as pd
from scipy.spatial.transform import Rotation as R
import yaml

ENABLE_VERBOSE = False

MAP_TAG_ID_TO_TAG_INDEX = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}

from camera_remap import (
    calculate_new_camera_matrix_and_remap_grids,
    read_and_preprocess_image )

def print_verbose(*args, **kwargs):
    global ENABLE_VERBOSE
    if ENABLE_VERBOSE:
        print(*args, **kwargs)

def print_non_verbose(*args, **kwargs):
    global ENABLE_VERBOSE
    if not ENABLE_VERBOSE:
        print(*args, **kwargs)

def read_calib(fn):
    calib_dict = dict()
    fs = cv2.FileStorage(fn, cv2.FILE_STORAGE_READ)
    
    calib_dict['tag_size'] = fs.getNode('tag_size').real()
    calib_dict['shape'] = {'H': 0, 'W': 0}
    calib_dict['shape']['H'] = int(fs.getNode('shape').getNode('H').real())
    calib_dict['shape']['W'] = int(fs.getNode('shape').getNode('W').real())
    calib_dict['poses'] = dict()
    node_poses = fs.getNode('poses')
    for i in range(5):
        calib_dict['poses'][f'T{i}'] = node_poses.getNode(f'T{i}').mat()
    calib_dict['num_tags'] = 4 # Assuming we have 4 tags.
    return calib_dict

def prepare_global_apriltag_corner_positions(calib_dict):
    # The canonical four corner points.
    tag_size = calib_dict['tag_size']
    half_tag_size = tag_size / 2
    
    # Homogeneous coordinates. 
    cannonical_corners = np.transpose( np.array(
        [ [ -half_tag_size, -half_tag_size, 0, 1 ],
          [  half_tag_size, -half_tag_size, 0, 1 ],
          [  half_tag_size,  half_tag_size, 0, 1 ],
          [ -half_tag_size,  half_tag_size, 0, 1 ] ] ) )

    global_corner_positions = []
    for i in range( calib_dict['num_tags'] ):
        pose = calib_dict['poses'][f'T{i}']
        global_corner_positions.append( pose @ cannonical_corners )
        
    return global_corner_positions

def read_cam_calib(fn):
    '''Kalibr format, only one camera.
    
    Assuming that the camera model is pinhole and the distortion model is radtan. 
    
    Returns:
    dict: Has keys "K" and "shape".
    '''
    
    with open(fn, 'r') as fp:
        calib_obj = yaml.load(fp, Loader=yaml.FullLoader)
    
    calib_obj = calib_obj['cam0']
    
    assert calib_obj['camera_model'] == 'pinhole', \
        f'Only support pinhole camera model. '\
        f'calib_obj["camera_model"] = {calib_obj["camera_model"]}. '
        
    assert calib_obj['distortion_model'] == 'radtan', \
        f'Only support radtan distortion model. '\
        f'calib_obj["distortion_model"] = {calib_obj["distortion_model"]}. '
        
    K = np.eye(3, dtype=np.float32)
    intrinsics = calib_obj['intrinsics']
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    
    distortion_coeffs = calib_obj['distortion_coeffs']
    D = np.array( distortion_coeffs, dtype=np.float32 )
    
    resolution = calib_obj['resolution']
    shape = (resolution[1], resolution[0]) # (H, W)
    
    return {
        'K': K,
        'D': D,
        'shape': shape }

def find_files(d):
    files = sorted(glob(join(d, 'frame_*.png'), recursive=False))
    assert len(files) > 0, f'No files found in {d}. '
    return files

def read_image(fn):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Failed to read {fn}. '
    return img

def get_image_shape(fn):
    img = read_image(fn)
    return img.shape[:2]

def convert_str_2_ints(s):
    ss = s.split(',')
    return [ int(x) for x in ss ]

def write_poses_2_csv(fn, poses_list):
    quat = [ pose[0] for pose in poses_list ]
    pos = [ pose[1] for pose in poses_list ]
    
    df_quat = pd.DataFrame(quat, columns=['qx', 'qy', 'qz', 'qw'])
    df_pos  = pd.DataFrame(pos, columns=['x', 'y', 'z'])
    
    df = pd.concat([df_quat, df_pos], axis=1)
    
    df.to_csv(fn, index=False, header=True)

def handl_args():
    global ENABLE_VERBOSE
    
    parser = argparse.ArgumentParser(\
        description=f'Estimate the camera poses from the apriltags detected in a sequence of '
                    f'images. ')
    parser.add_argument('--in_dir', type=str, required=True, 
                        help='The input directory. ')
    parser.add_argument('--apriltag_calib_fn', type=str, required=True, 
                        help='The calibration YAML file. ')
    parser.add_argument('--cam_calib_fn', type=str, required=True,
                        help='The camera calibration file, in Kalibr format. ')
    parser.add_argument('--new_img_shape', type=str, default='512,910',
                        help='The new image shape. ')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Show more information. ')
    args = parser.parse_args()
    
    if args.verbose:
        ENABLE_VERBOSE = True
        
    return args

if __name__ == '__main__':
    # Handle the arguments.
    args = handl_args()
    
    # Read the calibration file.
    calib_dict = read_calib(args.apriltag_calib_fn)
    print_verbose(calib_dict)
    
    # Prepare the global corner positions of the 4 apriltags.
    apriltag_global_corner_positions = prepare_global_apriltag_corner_positions(calib_dict)
    for i, pose in enumerate(apriltag_global_corner_positions):
        print_verbose(f'Apriltag {i}: \n{pose}')    
    
    # Read the camera calibration file.
    cam_calib = read_cam_calib(args.cam_calib_fn)
    print_verbose(f'Camera calibration: \n{cam_calib}')
    
    # Find the input images.
    files = find_files(args.in_dir)
    
    # Read the first image and figure out the image shape.
    img_shape = get_image_shape(files[0])
    
    assert img_shape == cam_calib['shape'], \
        f'Image shape and camera shape mismatch. '\
        f'img_shape = {img_shape}, cam_calib["shape"] = {cam_calib["shape"]}. '
    
    # The new image shape.
    new_img_shape = convert_str_2_ints(args.new_img_shape)
    
    # Get the new camera matrix regarding the new image shape.
    new_cam_matrix, map0, map1 = calculate_new_camera_matrix_and_remap_grids(
        cam_calib['shape'], new_img_shape, cam_calib['K'], cam_calib['D'] )
    print_verbose(f'new_img_shape = {new_img_shape}')
    print_verbose(f'new_cam_matrix = \n{new_cam_matrix}')
    
    new_cam_dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    poses_list = [] # Will be a list of 2-element lists. [ [ quat, pos ] ]
    
    # Create detector object
    detector = apriltag.Detector()
    
    for fn in files:
        print_verbose(fn)
        print_non_verbose('.', end='', flush=True)
        
        # Read and preprocess the image.
        img = read_and_preprocess_image(fn, map0, map1)
        
        # Convert the image to grayscale if it is a color image
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Detect AprilTags in the image
        detections = detector.detect(img)
        
        assert len(detections) > 0, f'No AprilTags detected in {fn}. '
        
        print_verbose(f'detection.tag_id = ', end='')
        
        detected_corners_2d = []
        detected_corners_3d = []
        for detection in detections:
            print_verbose(f'{detection.tag_id}, ', end='')
            
            tag_idx = MAP_TAG_ID_TO_TAG_INDEX[int(detection.tag_id)]
            
            tag_global_corners = np.transpose(
                apriltag_global_corner_positions[tag_idx][:3, :] )
            
            detected_corners_3d.append( tag_global_corners )
            
            # Get the 2D points of the Apriltag corners in pixels.
            corners2d = np.array([
                detection.corners[0],
                detection.corners[1],
                detection.corners[2],
                detection.corners[3],
            ])
            
            detected_corners_2d.append( corners2d )
        
        print_verbose('')
        
        detected_corners_2d = np.concatenate(detected_corners_2d, axis=0)
        detected_corners_3d = np.concatenate(detected_corners_3d, axis=0)
            
        success, rvec, tvec = cv2.solvePnP(
            detected_corners_3d, 
            detected_corners_2d, 
            new_cam_matrix, 
            new_cam_dist_coeffs)

        if not success:
            raise Exception('Camera pose estimation failed. ')

        # Convert rotation vector to rotation matrix
        rot_mat, _ = cv2.Rodrigues(rvec)
        
        rot_mat = np.transpose(rot_mat)
        tvec = -rot_mat @ tvec
        
        q = R.from_matrix(rot_mat).as_quat()
        
        print_verbose(f'q = {q}, t = {tvec.flatten()}')
        
        poses_list.append( [ q, tvec.flatten() ] )
        
    print_non_verbose('')
        
    out_fn = join( args.in_dir, 'poses.csv' )
    write_poses_2_csv(out_fn, poses_list)
    print(f'Poses written to {out_fn}. ')
    
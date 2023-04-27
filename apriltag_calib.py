
# This is from ChatGPT....

import apriltag
import argparse
import copy
import cv2
import glob
import numpy as np
import os
# import yaml

def test_directory_from_filename(fn):
    d = os.path.dirname(fn)
    os.makedirs(d, exist_ok=True)

def find_files(d, ext='.jpg'):
    files = sorted( glob.glob(
        os.path.join(d, f'*{ext}'), recursive=False ) )
    
    assert len(files) > 0, f'No files found in {d}. By pattern *{ext}. '
    
    return files

def calculate_new_camera_matrix_and_remap_grids(
    original_shape, 
    new_shape, 
    camera_matrix, 
    distortion_coeffs):
    
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coeffs,
        imageSize=( original_shape[1], original_shape[0] ),
        alpha=0.0,
        newImgSize=( new_shape[1], new_shape[0] ) )
    
    map0, map1 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coeffs,
        R=None,
        newCameraMatrix=new_cam_matrix,
        size=( new_shape[1], new_shape[0] ),
        m1type=cv2.CV_32FC1)
    
    return new_cam_matrix, map0, map1

def get_video_writer(out_fn, fps, frame_shape):
    '''
    frame_shape: (height, width)
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(out_fn, fourcc, fps, frame_shape[::-1])

def read_and_preprocess_image(image_path, map0, map1):
    """
    Reads an image and resizes it to 640x360.
    :param image_path: Path to the image file.
    :param map0: NumPy arrah, the map grid needed by cv2.remap.
    :param map1: NumPy arrah, the map grid needed by cv2.remap.
    :return: Preprocessed image.
    """
    # Load the image as-is
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    assert img is not None, f'Could not read image from {image_path}. '

    # Resize the image to 640x360
    # img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
    img = cv2.remap(img, map0, map1, interpolation=cv2.INTER_LINEAR)

    return img

def detect_poses(img, camera_matrix, tag_size=0.2):
    """
    Detects AprilTags in the given image and estimates the camera pose for each detected tag.
    :param img: The input image. Needs to be color.
    :param camera_matrix: The camera intrinsics matrix. A 3x3 numpy array.
    :param tag_size: Size of the AprilTag in meters. Default is 0.2 meters.
    :return: List of R (3x3 rotation matrix), t (3x1 translation vector) tuples for each detected tag
    """

    assert img.ndim == 3 and img.shape[2] == 3, \
        f'Input image must be grayscale. Got img.shape = {img.shape}. '

    # Make a copy of img.
    img_color = copy.deepcopy(img)
    
    # Convert the image to grayscale if it is a color image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create detector object
    detector = apriltag.Detector()

    # Detect AprilTags in the image
    detections = detector.detect(img)

    # For each detected tag, estimate camera pose and draw boundary and corner points
    poses = []
    tag_ids = []
    detected_corners_2d = []
    for detection in detections:
        # Append the tag ID to the list of tag IDs.
        tag_ids.append(detection.tag_id)
        
        # Get the 3D points of the AprilTag corners in meters
        corners3d = np.array([
            [-tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, tag_size / 2, 0],
            [-tag_size / 2, tag_size / 2, 0],
        ])

        # Get the 2D points of the AprilTag corners in pixels
        corners2d = np.array([
            detection.corners[0],
            detection.corners[1],
            detection.corners[2],
            detection.corners[3],
        ])
        
        detected_corners_2d.append(corners2d)

        # Estimate camera pose from 3D-2D point correspondences
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(corners3d, corners2d, camera_matrix, dist_coeffs)

        if not success:
            raise Exception('Camera pose estimation failed. ')

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Draw boundary and corner points of the detected tag
        boundary_color     = (0, 0, 255)
        conner_point_color = (0, 255, 0)
        tag_id_color       = (37, 190, 67)
        
        cv2.polylines(img_color, [np.int32(corners2d)], True, boundary_color, 2)
        for i, corner in enumerate(corners2d):
            integer_corner = tuple(np.int32(corner))
            cv2.putText(img_color, 
                        str(i), 
                        integer_corner, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        conner_point_color, 
                        2)

        # Draw the tag ID.
        center_point = tuple(np.int32(np.mean(corners2d, axis=0)))
        cv2.putText(img_color, 
                    str(detection.tag_id), 
                    center_point, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    tag_id_color, 
                    2)

        # Append the rotation matrix and translation vector to the list of poses
        poses.append((R, tvec))

    # Return the list of detected poses
    return tag_ids, detected_corners_2d, poses, img_color

def chain_pose(Rt_in_cam_0, Rt_in_cam_1):
    R0, t0 = Rt_in_cam_0
    R1, t1 = Rt_in_cam_1
    R0_T = np.transpose(R0)
    return R0_T @ R1, R0_T @ (t1 - t0)

def assemble_R_t(R, t):
    '''Assemble a 4x4 transformation matrix from R and t.
    '''
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t.flatten()
    return T

def pose_chain_2_abs_poses(pose_chain):
    '''
    pose_chain is a dict. The key is a string of the following format:
    "id0-id1"
    where id0 < id1. The value is a 4 x 4 matrix, representing the pose of id1 w.r.t. id0, 
    measured in the frame of id0.
    
    It is assumed that the pose_chain contains a chain of poses from index 0. Then this function
    queries the pose chain for the pose pairs 0-1, 1-2, ... one by one. The output yaml file 
    records the poses of all the nodes in the chain w.r.t. index 0 whose pose is the origin.
    '''
    
    # Get the total number of pairs in the pose chain.
    n_pairs = len(pose_chain)
    
    poses = [ np.eye(4) ]
    
    # Query the pose chain for the pose pairs 0-1, 1-2, ... one by one.
    for i in range(n_pairs):
        chain_id = f'{i}-{i+1}'
        rel_pose = pose_chain[chain_id]
        abs_pose = poses[i]
        poses.append( abs_pose @ rel_pose )
        
    return poses

# def dump_pose_chain_to_yaml_file(out_file, pose_chain):
#     '''Dump pose chain to a yaml file.
#     '''
    
#     poses = pose_chain_2_abs_poses(pose_chain)
    
#     pose_dicts = {
#         'poses': [ {'id': i, 'T': pose.tolist()} for i, pose in enumerate(poses) ] }
    
#     with open(out_file, 'w') as f:
#             yaml.dump(pose_dicts, f)
            
#     return pose_dicts

def dump_pose_chain_to_yaml_file(out_file, pose_chain, new_shape, tag_size):
    '''Dump pose chain to a yaml file.
    '''
    
    poses = pose_chain_2_abs_poses(pose_chain)
    
    fs = cv2.FileStorage(out_file, cv2.FILE_STORAGE_WRITE)
    
    fs.write('tag_size', tag_size)
    
    fs.startWriteStruct('shape', cv2.FileNode_MAP)
    fs.write('H', new_shape[0])
    fs.write('W', new_shape[1])
    fs.endWriteStruct()
    
    fs.startWriteStruct('poses', cv2.FileNode_MAP)
    
    for i, pose in enumerate(poses):
        fs.write(f'T{i}', pose)
    
    fs.endWriteStruct()
    
    fs.release()
    
    pose_dicts = {
        'poses': [ {'id': i, 'T': pose.tolist()} for i, pose in enumerate(poses) ] }
            
    return pose_dicts

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Estimate camera pose from AprilTag detection')
    parser.add_argument('--in-dir', type=str, 
                        help='Directory of the input images. ')
    parser.add_argument('--out-dir', type=str, 
                        help='The output directory. ')
    parser.add_argument('--ext', type=str, default='.jpg',
                        help='The extension of the images inlcuding the dot (default: .jpg). ')
    parser.add_argument('--tag-size', type=float, default=0.0865, 
                        help='Size of the AprilTag in meters (default: 0.2)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # The original camera intrinsics, before scaling, and undistortion.
    fx = 1753.19817998
    fy = 1756.72578275
    cx = 942.27459981
    cy = 545.93365452
    distortion_coeffs = np.array([-0.17963482, 0.16494141, -0.00117621, 0.00011371])

    camera_matrix = np.array([
            [ fx,  0, cx ],
            [  0, fy, cy ],
            [  0,  0,  1 ],
        ])
    
    # Get the remap matrices and the new camera matrix.
    original_shape = ( 1080, 1920 ) # H, W.
    # new_shape = (360, 640) # H, W.
    new_shape = (512, 910) # H, W.
    new_cam_matrix, map0, map1 = calculate_new_camera_matrix_and_remap_grids(
        original_shape, new_shape, camera_matrix, distortion_coeffs )
    
    print(f'new_cam_matrix = \n{new_cam_matrix}')

    # Find all images in the input directory.
    files = find_files(args.in_dir, ext=args.ext)
    
    # Prepare the calibration output.
    # pose_chain is a dict. The key is a string of the following format:
    # "id0-id1"
    # where id0 < id1. The value is a 4 x 4 matrix, representing the pose of id1 w.r.t. id0, 
    # measured in the frame of id0.
    pose_chain = dict()
    
    for fn in files:
        print(fn)
        
        # Read and preprocess the image.
        img = read_and_preprocess_image(fn, map0, map1)
        
        # Detect AprilTags in the image and estimate the camera pose for each tag.
        tag_ids, detected_corners_2d, poses, img = \
            detect_poses(img, camera_matrix=new_cam_matrix, tag_size=args.tag_size)
        
        assert len(tag_ids) == 2, f'len(tag_ids) = {len(tag_ids)}. '

        # Change 3-0 pair to 3-4 pair.
        if 0 in tag_ids and 3 in tag_ids:
            if tag_ids[0] == 0:
                tag_ids[0] = 4
            else:
                tag_ids[1] = 4

        if tag_ids[0] < tag_ids[1]:
            chained_pose = chain_pose(poses[0], poses[1])
            chain_id = f'{tag_ids[0]}-{tag_ids[1]}'
        else:
            chained_pose = chain_pose(poses[1], poses[0])
            chain_id = f'{tag_ids[1]}-{tag_ids[0]}'

        assert chain_id not in pose_chain, f'chain_id = {chain_id} already exists. '
        pose_chain[chain_id] = assemble_R_t(*chained_pose)

        for id, pose in zip(tag_ids, poses):
            print(f'Tag ID: {id}')
            print('Pose: ')
            print(f'R = \n{pose[0]}')
            print(f't = \n{pose[1]}')

        print(f'Chained pose of {chain_id}: ')
        print(f'T = \n{pose_chain[chain_id]}')

        # Show the image with detected tags
        cv2.imshow("AprilTags", img)
        
        out_fn = os.path.join(args.out_dir, os.path.basename(fn))
        cv2.imwrite(out_fn, img)
        
        k = cv2.waitKey(100)
        # Wait for the Q key to be pressed
        if k == ord('q'):
            print('Q key pressed. Quitting. ')
            break
    
    cv2.destroyAllWindows()
    
    # Save the pose chain to a YAML file.
    out_fn = os.path.join(args.out_dir, 'poses.yaml')
    pose_dicts = dump_pose_chain_to_yaml_file(out_fn, pose_chain, new_shape, tag_size=args.tag_size)
    
    for pose_dict in pose_dicts['poses']:
        print(f'{pose_dict["id"]}.T = \n{pose_dict["T"]}')
        

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def read_calib_result(fn):
    fs = cv2.FileStorage(fn, cv2.FILE_STORAGE_READ)
    assert fs, f'Failed to open {fn}. '
    
    # poses is a mapping.
    poses = fs.getNode('poses')
    
    pose_dicts = [
        { 'id': i, 'T':poses.getNode(f'T{i}').mat() } for i in range(poses.size()) ]
    
    # The tag_size.
    tag_size = fs.getNode('tag_size').real()
    
    calib = {
        'tag_size': tag_size,
        'pose_dicts': pose_dicts }
    
    return calib

def base_corner_positions(tag_size_meter):
    half = tag_size_meter / 2
    return np.transpose( np.array([
        [ -half, -half, 0 ],
        [  half, -half, 0 ],
        [  half,  half, 0 ],
        [ -half,  half, 0 ] ] ) )
    
def draw_pose_dicts(out_fn, pose_dicts, tag_size_meter):
    # The additional transform from the camera frame to the plot frame.
    T_plot_cam = np.eye(4)
    T_plot_cam[1, 1] = -1
    T_plot_cam[2, 2] = -1
    
    # The base position of the corner points.
    base_corners = base_corner_positions(tag_size_meter)
    
    # Homogeneous coordinates. 
    base_corners = np.concatenate([base_corners, np.ones((1, base_corners.shape[1]))], axis=0)
    
    fig, axes = plt.subplots()
    
    for pose_dict in pose_dicts:
        id = pose_dict['id']
        T = pose_dict['T']
        
        T = T_plot_cam @ T
        
        # Transform the base corner points.
        tag_corners = T @ base_corners
        
        for i in range(tag_corners.shape[1]):
            axes.add_artist(plt.Circle(tag_corners[:2, i], 0.01, color='r', fill=False))
            
        plt.text( T[0, 3], T[1, 3], f'{id}', fontsize=12)
    
    # plt.autoscale()
    axes.set_aspect( 1 )
    axes.set_xlim(-0.1, 0.5)
    axes.set_ylim(-0.5, 0.1)
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    axes.set_title('Calibrated Apriltag corner points.')
    
    # Save.
    fig.savefig(out_fn, dpi=300)
    
    plt.show()

def handle_args():
    parser = argparse.ArgumentParser(description='Draw the calibrated apriltag corner points. ')
    
    parser.add_argument('--calib-fn', type=str, 
                        help='The calibration result file in YAML format. ')
    
    parser.add_argument('--out-fn', type=str,
                        help='The output image filename. ')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = handle_args()
    
    os.makedirs(os.path.dirname(args.out_fn), exist_ok=True)
    
    calib = read_calib_result(args.calib_fn)
    pose_dicts = calib['pose_dicts']
    
    for pose_dict in pose_dicts:
        print(f'id = {pose_dict["id"]} \nT = {pose_dict["T"]}')
        
    draw_pose_dicts(args.out_fn, pose_dicts, calib['tag_size'])

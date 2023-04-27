
# This script helps to visualize the poses saved in a CSV file.

import argparse
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, dirname
import pandas as pd
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.rotations import matrix_from_quaternion
from pytransform3d.transformations import transform_from
from pytransform3d.transform_manager import TransformManager

plt.rcParams['figure.figsize'] = [10, 10]

def read_csv_as_df(fn):
    return pd.read_csv(fn, header=0)

def handle_args():
    parser = argparse.ArgumentParser(description='Visualize the poses. ')
    
    parser.add_argument('--pose_fn', type=str, required=True, 
                        help='The CSV file containing the poses. ')
    parser.add_argument('--show_plot', action='store_true', default=False, 
                        help='Set this flag to enable showing the plot on the screen. ')
    parser.add_argument('--title', type=str, required=True, 
                        help='The title of the plot. ')
    return parser.parse_args()

if __name__ == '__main__':
    args = handle_args()
    
    tm = TransformManager()
    poses_df = read_csv_as_df(args.pose_fn)
    for index, row in poses_df.iterrows():
        R = matrix_from_quaternion( (row['qw'], row['qx'], row['qy'], row['qz']) )
        t = np.array([ row['x'], row['y'], row['z'] ])
        tm.add_transform(f'{index:03d}', 'world', transform_from(R, t))

    ax = tm.plot_frames_in('world', s=0.05)
    ax.set_xlim((-0.5, 1.0))
    ax.set_ylim((-0.5, 1.0))
    ax.set_zlim((-1.0, 0.0))
    ax.view_init(elev=-90, azim=0, roll=-90)
    ax.set_title(args.title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    
    # Save the figure to a file.
    out_fig = join(dirname(args.pose_fn), 'poses_vis.png')
    plt.savefig(out_fig, bbox_inches='tight')
    
    if args.show_plot:
        plt.show()
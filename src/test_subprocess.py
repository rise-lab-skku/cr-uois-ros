#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import rospy
import rospkg


if __name__ == '__main__':
    rospy.init_node('test_subprocess', anonymous=True)

    # get ros package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('uois_ros')

    # get python path
    python_path = os.path.join(package_path, 'src', 'ros_util', 'roi_polygon_mask_generator.py')

    # get config path
    color_image_path = os.path.join(package_path, 'config', 'color_image.npy')
    roi_mask_path = os.path.join(package_path, 'config', 'roi_mask.npy')

    # run python script
    subprocess.call([
        'python', python_path,
        '--color_image_path', color_image_path,
        '--roi_mask_image_path', roi_mask_path])

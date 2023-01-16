#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess

import numpy as np

import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from uois_ros.srv import InitSegmask, GetSegmask

import data_augmentation as data_augmentation
import segmentation as segmentation


def imgmsg_to_cv2(img_msg):
    """Convert ROS Image messages to OpenCV images.

    `cv_bridge.imgmsg_to_cv2` is broken on the Python3.
    `from cv_bridge.boost.cv_bridge_boost import getCvType` does not work.

    Args:
        img_msg (`sonsor_msgs/Image`): ROS Image msg

    Raises:
        NotImplementedError: Supported encodings are "8UC3" and "32FC1"

    Returns:
        `numpy.ndarray`: OpenCV image
    """
    # check data type
    if img_msg.encoding == '8UC3':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == 'bgr8':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == 'rgb8':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == '32FC1':
        dtype = np.float32
        n_channels = 1
    elif img_msg.encoding == '32FC3':
        dtype = np.float32
        n_channels = 3
    elif img_msg.encoding == '64FC1':
        dtype = np.float64
        n_channels = 1
    else:
        raise NotImplementedError('custom imgmsg_to_cv2 does not support {} encoding type'.format(img_msg.encoding))

    # bigendian
    dtype = np.dtype(dtype)
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    if n_channels == 1:
        img = np.ndarray(shape=(img_msg.height, img_msg.width),
                         dtype=dtype, buffer=img_msg.data)
    else:
        img = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                         dtype=dtype, buffer=img_msg.data)

    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        img = img.byteswap().newbyteorder()
    return img


def cv2_to_imgmsg(img, encoding):
    """Convert an OpenCV image to a ROS Image message.

    `cv_bridge.imgmsg_to_cv2` is broken on the Python3.
    `from cv_bridge.boost.cv_bridge_boost import getCvType` does not work.

    Args:
        img (`numpy.ndarray`): OpenCV image
        encoding (str): Encoding of the image.

    Raises:
        NotImplementedError: Supported encodings are "8UC3" and "32FC1"

    Returns:
        `sensor_msgs/Image`: ROS Image msg
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('img must be of type numpy.ndarray')

    # check encoding
    if encoding == "passthrough":
        raise NotImplementedError('custom cv2_to_imgmsg does not support passthrough encoding type')

    # create msg
    img_msg = Image()
    img_msg.height = img.shape[0]
    img_msg.width = img.shape[1]
    img_msg.encoding = encoding
    if img.dtype.byteorder == '>':
        img_msg.is_bigendian = True
    img_msg.data = img.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg


class ROIPolygonMaskGenerator(object):
    """Generate ROI mask from polygon ROI.

    This class is a wrapper of the python script
    `uois_ros/src/ros_util/roi_polygon_mask_generator.py`.
    Reason of this wrapper is that matplotlib is not compatible with python3
    and ROS melodic.
    """
    def __init__(self):
        pkg_path = rospkg.RosPack().get_path('uois_ros')
        self.python_path = pkg_path + '/src/ros_util/roi_polygon_mask_generator.py'
        self.roi_mask_image_path = pkg_path + '/config/roi_mask_image.npy'
        self.color_image_path = pkg_path + '/config/color_image.npy'

    def generate_mask(self, color_image):
        # save color image
        np.save(self.color_image_path, color_image)

        # call python script to generate mask
        subprocess.call([
            'python', self.python_path,
            '--color_image_path', self.color_image_path,
            '--roi_mask_image_path', self.roi_mask_image_path])

        # delete color image
        os.remove(self.color_image_path)

        # load mask
        self.load_mask()

    def load_mask(self):
        mask = np.load(self.roi_mask_image_path)
        self.mask = mask


class SegmentationServer(object):
    def __init__(self):
        # Load config
        dsn_config = rospy.get_param('~dsn_config')
        rrn_config = rospy.get_param('~rrn_config')
        uois3d_config = rospy.get_param('~uois3d_config')

        # Get checkpoint path
        rospack = rospkg.RosPack()
        pack_path = rospack.get_path('uois_ros')
        checkpoint_dir = os.path.join(pack_path, 'checkpoints')

        # Load model
        dsn_filename = os.path.join(
            checkpoint_dir,
            'DepthSeedingNetwork_3D_TOD_checkpoint.pth')
        rrn_filename = os.path.join(
            checkpoint_dir,
            'RRN_OID_checkpoint.pth')
        uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
        self.uois_net_3d = segmentation.UOISNet3D(
            uois3d_config,
            dsn_filename,
            dsn_config,
            rrn_filename,
            rrn_config)

        # Init roi mask generator
        self.roi_mask_gen = ROIPolygonMaskGenerator()
        try:
            self.roi_mask_gen.load_mask()
        except Exception:
            rospy.logwarn('Failed to load ROI mask. Please initialize segmentation parameters.')

        # initialize ros server
        rospy.Service(
            '~init_segmask',
            InitSegmask,
            self.init_segmask_handler)
        rospy.Service(
            '~get_segmask',
            GetSegmask,
            self.get_segmask_handler)

        # cv_bridge
        self.cv_bridge = CvBridge()

        # print
        rospy.loginfo('UIOS server start')

    def init_segmask_handler(self, req):
        """ ROS service handler for `~init_segmask` service.

        It will generate ROI mask from user input.

        Args:
            req (`uois_ros.srv.InitSegmask`): ROS service request
        """
        try:
            rospy.loginfo('Start segmentation initialization(ROI mask generation)...')
            color_image = imgmsg_to_cv2(req.color_image)
            self.roi_mask_gen.generate_mask(color_image)
            rospy.loginfo('Succeed to initialize segmentation parameters')
            return True
        except rospy.ServiceException('Failed to initialize segmentation parameters'):
            return False

    def request_to_batch(self, req):
        """Convert ROS Image message to batch.

        Args:
            req (`uois_ros.srv.GetSegmask`): ROS service request

        Returns:
            `dict`: batch
        """
        # convert image
        raw_rgb_img = imgmsg_to_cv2(req.rgb_image)
        raw_xyz_img = imgmsg_to_cv2(req.xyz_image)

        im_height, im_width, _ = raw_rgb_img.shape
        if im_height > 480 or im_width > 640:
            rospy.logerr('Image size is too large. Image size must be less than 480x640. But {}x{}'.format(im_height, im_width))

        # make empty batch
        rgb_imgs = np.zeros((1, 480, 640, 3), dtype=np.float32)
        xyz_imgs = np.zeros((1, 480, 640, 3), dtype=np.float32)

        # fill small image to center of 480, 640 image
        h_offset = (480 - im_height) // 2
        w_offset = (640 - im_width) // 2
        rgb_imgs[0, h_offset:h_offset + im_height, w_offset:w_offset + im_width, :] = raw_rgb_img
        xyz_imgs[0, h_offset:h_offset + im_height, w_offset:w_offset + im_width, :] = raw_xyz_img

        # standardize image
        rgb_imgs[0] = data_augmentation.standardize_image(rgb_imgs[0])

        # make batch
        batch = {
            'rgb': data_augmentation.array_to_tensor(rgb_imgs),
            'xyz': data_augmentation.array_to_tensor(xyz_imgs),
        }
        return batch, im_height, im_width

    def get_segmask_handler(self, req):
        """ROS service handler for `~get_segmask` service.

        Args:
            req (`uois_ros.srv.GetSegmask`): ROS service request

        Returns:
            `uois_ros.srv.GetSegmaskResponse`: ROS service response. Segmask image type is `uint16`.
        """
        try:
            start_time = rospy.Time.now()

            # get segmentation mask from uios network
            rospy.loginfo('Get segmentation request from the client.')
            batch, im_height, im_width = self.request_to_batch(req)
            _, _, _, segmasks = self.uois_net_3d.run_on_batch(batch)
            segmasks = segmasks.cpu().numpy()
            uois_segmask = segmasks[0]

            # crop segmentation mask
            h_offset = (480 - im_height) // 2
            w_offset = (640 - im_width) // 2
            uois_segmask = uois_segmask[h_offset:h_offset + im_height, w_offset:w_offset + im_width]

            # convert to ROS message
            uois_segmask = uois_segmask.astype(np.uint16)

            # get roi mask
            roi_segmask = self.roi_mask_gen.mask
            rospy.logwarn('roi_segmask: {}'.format(roi_segmask.shape))
            rospy.logwarn('roi_segmask: {}'.format(roi_segmask.dtype))

            # combine roi mask and uois mask
            # reverse roi mask(bool)
            roi_segmask = np.logical_not(roi_segmask)
            uois_segmask[roi_segmask] = 0
            seg_mask_msg = cv2_to_imgmsg(uois_segmask, encoding='16UC1')

            rospy.loginfo('Succeed to get segmentation mask. Elapsed time: {}[s]'.format((rospy.Time.now() - start_time).to_sec()))
            return seg_mask_msg
        except rospy.ServiceException('Failed to get segmentation mask'):
            return Image()


if __name__ == '__main__':
    rospy.init_node('uois_ros_node')
    s = SegmentationServer()
    rospy.spin()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches


class ROIPolygonMaskGenerator(object):
    def set_roi_polygon(self, image):
        """Set RoI polygon.

        Args:
            image (np.ndarray): color image.
        """
        print('Set RoI polygon.')
        # get image
        self.image = image

        # init plots
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title('ROI Polygon: Click mouse to add point / Press `space bar` to delete point')

        # init polygon
        self.polygon = ax.add_patch(plt_patches.Polygon(np.array([[0, 0]]), facecolor=[1,0,0, 0.5], edgecolor=[0,0,0]))
        self.xy = self.polygon.get_xy()

        # connect events
        self.cid_mouse = self.polygon.figure.canvas.mpl_connect(
            'button_press_event', self.on_mouse_event)
        self.cid_key = self.polygon.figure.canvas.mpl_connect(
            'key_press_event', self.on_key_event)

        # flags
        self.cid_mouse_init_flag = False

        plt.show()

    def on_mouse_event(self, event):
        """Add new point to polygon."""
        if event.inaxes != self.polygon.axes:
            return

        # get new xy data
        new_xy = np.array([[event.xdata, event.ydata]])
        print('Add new point: {}'.format(new_xy))

        # at first delete temp point and add new point
        if self.cid_mouse_init_flag is False:
            self.xy = new_xy
            self.cid_mouse_init_flag = True
        else:
            self.xy = np.r_[self.xy, new_xy]
        self.polygon.set_xy(self.xy)
        self.polygon.figure.canvas.draw()

    def on_key_event(self, event):
        """Delete last point from polygon.

        Args:
            event (matplotlib.backend_bases.KeyEvent): key event.
        """
        if event.inaxes != self.polygon.axes:
            return
        if event.key == ' ':
            if len(self.xy) == 1:
                self.cid_mouse_init_flag = False
            else:
                p = self.xy[-1]
                self.xy = self.xy[:-1]
                print('Delete point: {}'.format(p))
        self.polygon.set_xy(self.xy)
        self.polygon.figure.canvas.draw()

    def get_mask(self):
        """Get RoI mask.

        Returns:
            mask (np.ndarray): boolean mask.
        """
        H, W, C = self.image.shape
        y, x = np.mgrid[:H, :W]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        path = self.polygon.get_path()
        mask = path.contains_points(coors)
        mask = mask.reshape(H, W)
        return mask

    def save_mask(self, roi_mask_image_path):
        """Save RoI mask.

        Args:
            roi_mask_image_path (str): RoI mask image path.
        """
        mask = self.get_mask()
        np.save(roi_mask_image_path, mask)
        self.mask = mask


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_image_path', type=str, default=None)
    parser.add_argument('--roi_mask_image_path', type=str, default=None)
    args = parser.parse_args()
    color_image_path = args.color_image_path
    roi_mask_image_path = args.roi_mask_image_path

    # check color image is exist for 5 seconds
    timeout = time.time() + 5
    while time.time() < timeout:
        if os.path.exists(color_image_path):
            break
        time.sleep(0.1)
    if not os.path.exists(color_image_path):
        raise Exception('Color image is not exist: {}'.format(color_image_path))

    # load color image
    color_image = np.load(color_image_path)

    # set roi polygon
    roi_polygon_mask_generator = ROIPolygonMaskGenerator()
    roi_polygon_mask_generator.set_roi_polygon(color_image)
    roi_polygon_mask_generator.save_mask(roi_mask_image_path)
    print('Save mask image: {}'.format(roi_mask_image_path))

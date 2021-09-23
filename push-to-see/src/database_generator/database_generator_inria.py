import socket
import select
import struct
import time
import os
import sys

import matplotlib
import numpy as np
# import utils
import torch


import matplotlib.pyplot as plt
import yaml
import cv2
from PIL import Image

import pylab as plt
import pandas as pd
import imageio

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, "../.."))
from simulation import vrep


class DatabaseGenerator(object):
    def __init__(self, config):

        # --class label convex info-- #

        obj_config = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/object_class.yaml'
        with open(obj_config) as f:
            self.obj_inf = yaml.load(f, Loader=yaml.FullLoader)
        self.convex_class = self.obj_inf['convex_class']
        self.total_convex = self.obj_inf['overview']['total_convex']

        print(self.convex_class)

        self.handle_objclass, self.pixel_class = dict(), dict()

        # --------------------------- #

        # --final result as mask rcnn target-- #
        self.mask_rcnn_target = []

        # should be empty after each save
        self.single_img_target = {'boxes': None, 'labels': None, 'masks': None}

        # ------------------------------------ #

        self.config = config

        self.DATABASE_SIZE = config['database']['settings']['database_size']
        self.NUM_OBJ_MAX = config['database']['settings']['max_num_obj']
        self.NUM_OBJ_MIN = config['database']['settings']['min_num_obj']
        # self.SELECTION_POOL = config['database']['settings']['total_num_obj']

        self.DROP_HEIGHT = config['database']['settings']['drop_height']
        self.SAVE_NUMPY = config['data']['save_numpy']
        self.SAVE_PNG = config['data']['save_png']
        self.SAVE_COLOR = config['data']['save_color_img']

        self.generated = config['database']['settings']['generated']

        self.drop_limits = np.asarray([[-0.6, -0.4], [-0.15, 0.15], [-0.2, -0.1]])

        # Make sure to have the server side running in V-REP:
        # in a child script of a V-REP scene, add following command
        # to be executed just once, at simulation start:
        #
        # simExtRemoteApiStart(19999)
        #
        # then start simulation, and run this program.
        #
        # IMPORTANT: for each successful call to simxStart, there
        # should be a corresponding call to simxFinish at the end!

        # MODIFY remoteApiConnections.txt

        # Connect to simulator
        vrep.simxFinish(-1)  # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        # Setup virtual camera in simulation
        # self._setup_sim_camera()

        # Get handle to camera
        # sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp',
        #                                                     vrep.simx_opmode_blocking)

        # Get handles for masking
        sim_ret_cam_gt, self.cam_handle_gt = vrep.simxGetObjectHandle(self.sim_client, 'gt_sensor',
                                                                      vrep.simx_opmode_blocking)
        print(f'sim_ret_cam_gt {sim_ret_cam_gt}, type {type(sim_ret_cam_gt)}')

        # Get handles for rgb
        sim_ret_cam_ortho, self.cam_handle_ortho = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_ortho',
                                                                            vrep.simx_opmode_blocking)

    # def _setup_sim_camera(self):

    # sim_ret_cam_mask, self.cam_handle_mask = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp0',
    #                                                                     vrep.simx_opmode_blocking)

    # Get camera pose and intrinsics in simulation
    # sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1,
    #                                                    vrep.simx_opmode_blocking)
    # sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1,
    #                                                          vrep.simx_opmode_blocking)
    # cam_trans = np.eye(4, 4)
    # cam_trans[0:3, 3] = np.asarray(cam_position)
    # cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
    # cam_rotm = np.eye(4, 4)
    # cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
    # self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
    # self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
    # self.cam_depth_scale = 1

    # Get background image
    # self.bg_color_img, self.bg_depth_img = self.get_camera_data()
    # self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def generate_database(self):
        #print(self.DATABASE_SIZE)
        err_trials = []
        self._folder_config()
        init_time = time.time()

        next_scene = self.generated + 1
        # for i in range(0, self.DATABASE_SIZE):
        while next_scene < self.DATABASE_SIZE:
            # empty memory from last scene
            self.handle_objclass, self.pixel_class = dict(), dict()
            session_start = time.time()
            # np.random.seed()
            curr_num_obj = np.random.random_integers(self.NUM_OBJ_MIN, self.NUM_OBJ_MAX)

            print('Scene no %06d - Number of objects in the current scene --->' % next_scene, curr_num_obj)
            ret = self._add_objects(curr_num_obj)

            if not ret[0] == -1:
                self.save_scene(next_scene)
                next_scene += 1
            else:
                print("ERROR: Current scene couldn't save!")
                err_trials.append(next_scene)

            np.save(self.scene_info_dir + "scene_info_%06d.npy" % next_scene, np.asarray(ret[1]))
            self.restart_sim()
            session_end = time.time() - session_start
            print('Elapsed time for this current scene --> {: .02f} seconds'.format(session_end),
                  'Total elapsed time by now --> {: .02f} seconds'.format(time.time() - init_time))
        # TODO proper logging
        np.savetxt(os.path.join(self.scene_info_dir, 'error_log.txt'), err_trials, fmt='%06d', delimiter=' ')

        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.25)
        vrep.simxFinish(self.sim_client)

    def save_scene(self, iteration):
        curr_rgb, curr_depth, seg_mask, mask_relation, infor_for_maskrcnn = self.get_camera_data()

        # print(mask_relation)

        # save rgb and depth images
        color_image = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_dir, 'color_image_%06d.png' % iteration), color_image)

        depth_image = np.round(curr_depth * 10000).astype(np.uint16)  # Save depth in 1e-4 meters
        np.save(os.path.join(self.depth_dir_numpy, 'depth_%06d.npy' % iteration), depth_image)
        cv2.imwrite(os.path.join(self.depth_dir_png, 'depth_image_%06d.png' % iteration), depth_image)

        # save segmentation masks
        # np.save(os.path.join(self.segmask_dir_numpy, 'segmask_%06d.npy' % iteration), seg_mask)
        # plt.imsave(os.path.join(self.segmask_dir_png, 'segmask_image_%06d.png' % iteration), seg_mask)

        np.save(os.path.join(self.segmask_dir_numpy, 'segmask_%06d.npy' % iteration), infor_for_maskrcnn)
        cv2.imwrite(os.path.join(self.segmask_dir_png, 'segmask_image_%06d.png' % iteration), seg_mask)


    def restart_sim(self):

        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.25)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.25)

    def _get_segmentation_mask(self, mask_cam_handle):

        ret, res, data = vrep.simxGetVisionSensorImage(self.sim_client, mask_cam_handle, 0,
                                                       vrep.simx_opmode_oneshot_wait)

        # print(f'what is res {res}')
        # [1024, 1024], resolution

        # list, len3145728
        # print(f'type of data {type(data)}, len {len(data)}')

        # print(f'what is ret {ret}')

        # (1024, 1024, 3)
        seg_mask_temp = np.reshape(data, (res[1], res[0], 3))
        # print(f'shape 1 {seg_mask_temp.shape}')
        # plt.imshow(seg_mask_temp.astype(np.uint8))
        # plt.show()

        # (1024, 1024, 1)
        seg_mask_temp = seg_mask_temp[:, :, :1]
        # print(f'shape 2 {seg_mask_temp.shape}')
        # plt.imshow(seg_mask_temp.astype(np.uint8))
        # plt.show()

        # (1024, 1024)
        seg_mask_temp = np.reshape(seg_mask_temp, (res[1], res[0]))
        # print(f'shape 3 {seg_mask_temp.shape}')
        seg_mask = seg_mask_temp.copy()

        # print(type(seg_mask))
        # plt.imshow(seg_mask.astype(np.uint8))
        # plt.show()

        # set the background pixels to 0
        elements, counts = np.unique(seg_mask, return_counts=True)
        background_index = np.argmax(counts)
        # print(f'background_index {background_index}')
        new_mask = np.where(seg_mask == elements[background_index], 0, seg_mask)
        # plt.imshow(new_mask.astype(np.uint8))
        # plt.show()


        # elements in objects[] are handle ids of objects in vrep
        objects = np.delete(elements, background_index)
        objects = np.sort(objects)

        print(f'handle_objclass: {self.handle_objclass}')

        # pixel value which is handle_id and label
        pix_lab = []

        for i in range(0, objects.size):
            # ----------------- #
            if objects[i] < -100:
                objects[i] = 256 + objects[i]

            label = self.obj_inf['class_label'][self.handle_objclass[objects[i]]]
            #print(label)
            pix_lab.append([objects[i], label])




            pixel_value = 255 - i
            obj_class = self.handle_objclass[objects[i]]
            self.pixel_class[pixel_value] = obj_class
            new_mask[np.where(seg_mask == objects[i])] = objects[i]

        # mask rcnn documents requires 64bit
        # labels (Int64Tensor[N]): the class label for each ground-truth box
        #labels = np.array(labels)
        #print(labels)
        # flip according rows
        new_mask = np.flip(new_mask, axis=0)

        print(pix_lab)

        saved_infor = np.array(pix_lab)

        mask_relation = pd.DataFrame.from_dict(self.pixel_class, orient='index')

        # print(f'pixel x class:{self.pixel_class}')
        # print(f'mask_relation{mask_relation}, type {type(mask_relation)}')

        return new_mask, mask_relation, saved_infor

    def get_camera_data(self):

        print("Collecting images and ground truth masks...")
        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle_ortho, 0,
                                                                       vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        # color_img = np.fliplr(color_img)
        color_img = np.flip(color_img, axis=0)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle_ortho,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        # depth_img = np.fliplr(depth_img)
        depth_img = np.flip(depth_img, axis=0)
        zNear = 0.01  # 0.01
        zFar = 0.5  # 10
        depth_img = depth_img * (zFar - zNear) + zNear

        # Get ground truth segmentation masks
        seg_mask, mask_relation, saved_infor = self._get_segmentation_mask(self.cam_handle_gt)

        return color_img, depth_img, seg_mask, mask_relation, saved_infor

    def _add_objects(self, num_obj_heap):

        # 18 INRIA objects x 4 = 72 (convex_0 to convex_71)
        isOscillating = 1
        print(f'num_obi_heap {num_obj_heap}')
        objects = np.random.choice(range(0, self.total_convex), num_obj_heap, replace=False)
        shapes = []
        objects_info = []
        print(f'add_objects object {objects}')

        for i in objects:
            _, handle_i = vrep.simxGetObjectHandle(self.sim_client, f'convex_{i}', vrep.simx_opmode_blocking)
            obj_class = self.convex_class[i]
            self.handle_objclass[handle_i] = obj_class
        print(f'handle_objclass in _add start{self.handle_objclass}')

        for object_idx in objects:

            drop_x = (self.drop_limits[0][1] - self.drop_limits[0][0]) * np.random.random_sample() + \
                     self.drop_limits[0][0]
            drop_y = (self.drop_limits[1][1] - self.drop_limits[1][0]) * np.random.random_sample() + \
                     self.drop_limits[1][0]

            object_position = [drop_x, drop_y, self.DROP_HEIGHT]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]

            shape_name = 'convex_{:d}'.format(object_idx)
            shapes.append(shape_name)
            objects_info.append([shape_name, object_position, object_orientation])

            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                                  'remoteApiCommandServer',
                                                                                                  vrep.sim_scripttype_childscript,
                                                                                                  'moveShape',
                                                                                                  [0],
                                                                                                  object_position + object_orientation,
                                                                                                  [shape_name],
                                                                                                  bytearray(),
                                                                                                  vrep.simx_opmode_blocking)
            # break if error
            if ret_resp == 8:
                print('Failed to add new objects to simulation.')
                time.sleep(0.5)
                return [-1, -1]

            # wait a tad before dropping the next object
            time.sleep(0.04)

        # wait 100ms after all objects were released
        time.sleep(0.1)

        # wait until the oscillating objects stopped
        # (the remote method (i.e. checkMotion) that is attached to the vrep script (simulation_random_datacollect.ttt)
        # checks the angular velocities of the dropped objects)

        """
        while isOscillating:
            ret_r, ret_i, ret_f, ret_s, ret_b = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',
                                                                            vrep.sim_scripttype_childscript,
                                                                            'checkMotion',
                                                                            [len(shapes)],
                                                                            [0.0],
                                                                            shapes,
                                                                            bytearray(),
                                                                            vrep.simx_opmode_blocking)
            time.sleep(0.05)
            isOscillating = ret_i[0]

        return [0, objects_info]
        """
        counter = 0
        while isOscillating:
            counter += 1
            if counter < 20:
                ret_r, ret_i, ret_f, ret_s, ret_b = vrep.simxCallScriptFunction(self.sim_client,
                                                                                'remoteApiCommandServer',
                                                                                vrep.sim_scripttype_childscript,
                                                                                'checkMotion',
                                                                                [len(shapes)],
                                                                                [0.0],
                                                                                shapes,
                                                                                bytearray(),
                                                                                vrep.simx_opmode_blocking)
                time.sleep(0.03)
                # print(ret_i)
                # wait responose from remote api
                while not ret_i:
                    time.sleep(0.03)
                    ret_r, ret_i, ret_f, ret_s, ret_b = vrep.simxCallScriptFunction(self.sim_client,
                                                                                'remoteApiCommandServer',
                                                                                vrep.sim_scripttype_childscript,
                                                                                'checkMotion',
                                                                                [len(shapes)],
                                                                                [0.0],
                                                                                shapes,
                                                                                bytearray(),
                                                                                vrep.simx_opmode_blocking)
                isOscillating = ret_i[0]


            else:

                ret_r, ret_i, ret_f, ret_s, ret_b = vrep.simxCallScriptFunction(self.sim_client,
                                                                                'remoteApiCommandServer',
                                                                                vrep.sim_scripttype_childscript,
                                                                                'setStatic',
                                                                                [len(shapes)],
                                                                                [0.0],
                                                                                shapes,
                                                                                bytearray(),
                                                                                vrep.simx_opmode_blocking)

                isOscillating = 0

        return [0, objects_info]

    def _folder_config(self):
        """Path configuration to save collected data"""

        # tilde expansion if necessary
        if self.config['database']['path'].find('~') == 0:
            database_dir = os.path.expanduser(self.config['database']['path'])
        else:
            database_dir = self.config['database']['path']

        if not os.path.exists(database_dir):
            os.mkdir(database_dir)
        else:
            print(database_dir)
            print("WARNING: Folder to save database is already exist, "
                  "if it contains old scene files with same name, they will be overwritten!")

        self.depth_dir_numpy = os.path.join(database_dir, 'depth_ims/NUMPY/')
        self.depth_dir_png = os.path.join(database_dir, 'depth_ims/PNG/')
        self.color_dir = os.path.join(database_dir, 'color_ims/')
        self.segmask_dir_numpy = os.path.join(database_dir, 'segmentation_masks/NUMPY/')
        self.segmask_dir_png = os.path.join(database_dir, 'segmentation_masks/PNG/')
        self.scene_info_dir = os.path.join(database_dir, 'scene_info/')

        sub_folder_list = [self.depth_dir_numpy, self.depth_dir_png, self.color_dir, self.segmask_dir_numpy,
                           self.segmask_dir_png, self.scene_info_dir]
        for folder in sub_folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)


if __name__ == "__main__":
    # Read configuration file
    config_file = os.getcwd() + '/database_config_inria.yaml'
    with open(config_file) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    dg = DatabaseGenerator(configuration)
    dg.generate_database()

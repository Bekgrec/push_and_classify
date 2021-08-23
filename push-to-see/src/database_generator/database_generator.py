import socket
import select
import struct
import time
import os
import numpy as np
import utils
from simulation import vrep
import matplotlib.pyplot as plt
import yaml
import cv2


class DatabaseGenerator(object):
    def __init__(self, config):
        self.config = config

        self.DATABASE_SIZE = config['database']['settings']['database_size']
        self.NUM_OBJ_MAX = config['database']['settings']['max_num_obj']
        self.NUM_OBJ_MIN = config['database']['settings']['min_num_obj']
        self.SELECTION_POOL = config['database']['settings']['total_num_obj']

        self.DROP_HEIGHT = config['database']['settings']['drop_height']

        self.SAVE_NUMPY = config['data']['save_numpy']
        self.SAVE_PNG = config['data']['save_png']
        self.SAVE_COLOR = config['data']['save_color_img']

        # self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        self.workspace_limits = np.asarray([[-0.65, -0.5], [-0.15, 0.15], [-0.2, -0.1]])
        # Define colors for object meshes (Tableau palette)
        self.color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                                       [89.0, 161.0, 79.0],  # green
                                       [156, 117, 95],  # brown
                                       [242, 142, 43],  # orange
                                       [237.0, 201.0, 72.0],  # yellow
                                       [186, 176, 172],  # gray
                                       [255.0, 87.0, 89.0],  # red
                                       [176, 122, 161],  # purple
                                       [118, 183, 178],  # cyan
                                       [255, 157, 167]]) / 255.0  # pink

        # Read files in object mesh directory

        self.obj_mesh_dir = os.path.join(os.getcwd(), config['meshes']['object_path'])
        # self.obj_mesh_dir = "/home/baris/Workspace/visual-pushing-grasping/objects/blocks"
        # self.obj_mesh_dir = 'objects/blocks'
        self.obj_mesh_color = self.color_space[np.asarray(range(10)) % 10, :]
        # self.mesh_list = os.listdir(self.obj_mesh_dir)

        # Randomly choose objects to add to scene
        # self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        # self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

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
        self._setup_sim_camera()

        self.floor_handle = 0
        self.plane_handle = 0

    def _setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp',
                                                            vrep.simx_opmode_blocking)

        sim_ret_cam_ortho, self.cam_handle_ortho = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_ortho',
                                                                            vrep.simx_opmode_blocking)
        # Get handles for masking
        sim_ret_cam_mask, self.cam_handle_mask = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp0',
                                                                            vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1,
                                                           vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1,
                                                                 vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        # self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        # self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def generate_database(self):
        err_trials = []
        self._folder_config()
        init_time = time.time()
        for i in range(0, self.DATABASE_SIZE):
            session_start = time.time()

            curr_num_obj = np.random.random_integers(self.NUM_OBJ_MIN, self.NUM_OBJ_MAX)
            print('Scene no %06d - Number of objects in the current scene --->' % i, curr_num_obj)
            ret = self._add_objects(curr_num_obj, i)

            if not ret[0] == -1:
                self.save_scene(i)
            else:
                print("ERROR: Current scene couldn't save!")
                err_trials.append(i)

            # TODO compare remove objects vs restart_sim
            #self._remove_objects(ret[0])
            np.save(self.scene_info_dir + "scene_info_%06d.npy" % i, ret[1])
            self.restart_sim()
            session_end = time.time() - session_start
            print('Elapsed time for this current scene --> {: .02f} seconds'.format(session_end),
                  'Total elapsed time by now --> {: .02f} seconds'.format(time.time() - init_time))
        # TODO proper logging
        np.savetxt(os.path.join(self.scene_info_dir, 'error_log.txt'), err_trials, fmt='%06d', delimiter=' ')
        vrep.simxFinish(self.sim_client)

    def save_scene(self, iteration):
        curr_rgb, curr_depth, seg_mask_persp, seg_mask_ortho = self.get_camera_data()

        # save rgb and depth images
        color_image = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_dir, 'color_image_%06d.png' % iteration), color_image)
        depth_image = np.round(curr_depth * 10000).astype(np.uint16) # Save depth in 1e-4 meters
        np.save(os.path.join(self.depth_dir_numpy, 'depth_%06d.npy' % iteration), depth_image)
        cv2.imwrite(os.path.join(self.depth_dir_png, 'depth_image_%06d.png' % iteration), depth_image)

        # save segmentation masks
        # np.save(os.path.join(self.segmask_dir_numpy, 'segmask_%06d.npy' % iteration), seg_mask_persp)
        # plt.imsave(os.path.join(self.segmask_dir_png, 'segmask_image_%06d.png' % iteration), seg_mask_persp)
        cv2.imwrite(os.path.join(self.segmask_dir_png, 'segmask_image_%06d.png' % iteration), seg_mask_persp)

    def _remove_objects(self, object_handles):

        ret_values = []
        for i in object_handles:
            ret = vrep.simxRemoveObject(self.sim_client, i, vrep.simx_opmode_blocking)
            ret_values.append(ret)
        if ret_values.count(0) != len(object_handles):
            print("WARNING: One or more items might not be removed!")

    def restart_sim(self):

        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.25)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.25)

    def _get_segmentation_mask(self, mask_cam_handle):

        # if self.plane_handle == 0 or self.floor_handle == 0:

        # Get handle ids of the floor and the plane (at the moment Plane --> 19, Floor --> 10).
        # The related pixels will be removed whilst calculating the ground segmentation
        # ret_plane, self.plane_handle = vrep.simxGetObjectHandle(self.sim_client, 'Plane', vrep.simx_opmode_blocking)
        #
        # ret_floor, self.floor_handle = vrep.simxGetObjectHandle(self.sim_client, 'ResizableFloor_5_25_visibleElement',
        #                                              vrep.simx_opmode_blocking)

            # # TODO fix this!
            # if not (ret_plane == 0 or ret_floor == 0):
        self.plane_handle = 19
        self.floor_handle = 10


        ret, res, data = vrep.simxGetVisionSensorImage(self.sim_client, mask_cam_handle, 0,
                                                      vrep.simx_opmode_blocking)
        time.sleep(0.01)

        seg_mask_temp = np.reshape(data, (res[1], res[0], 3))
        # seg_mask = np.delete(seg_mask, np.s_[1:3], axis=2)
        seg_mask_temp = seg_mask_temp[:, :, :1]
        seg_mask_temp = np.reshape(seg_mask_temp, (res[1], res[0]))
        # When the robot is in the scene 88 Plane, 10 Floors(corners), 0 2nd and 3rd channel
        # Currently --> 'Plane' --> 19 'ResizableFloor_5_25_visibleElement' --> 10
        objects_ids = np.unique(seg_mask_temp)
        #TODO this is dangerous, remove this
        objects = np.delete(objects_ids, np.where(np.logical_or(objects_ids == self.plane_handle,
                                                                objects_ids == self.floor_handle)))

        seg_mask = seg_mask_temp.copy()
        seg_mask[np.where(np.logical_or(seg_mask_temp == self.plane_handle, seg_mask_temp == self.floor_handle))] = 0
        # segmentation_all = np.zeros((res[0], res[1], objects.size), dtype=np.uint8)
        # temp = seg_mask.copy()
        for i in range(0, objects.size):
            # segmentation_all[:, :, i][np.where(temp == objects[i])] = i + 1
            seg_mask[np.where(seg_mask == objects[i])] = i + 1

        seg_mask = np.flip(seg_mask, axis=1)
        return seg_mask

    def get_camera_data(self):

        print("Collecting images and ground truth masks...")
        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0,
                                                                       vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        # Get ground truth segmentation masks
        seg_mask_persp = self._get_segmentation_mask(self.cam_handle_mask)

        # seg_mask_ortho = self._get_segmentation_mask(self.cam_handle_ortho)
        seg_mask_ortho = -1

        return color_img, depth_img, seg_mask_persp, seg_mask_ortho

    def _add_objects(self, num_obj, iteration_no):

        object_handles = []
        object_order = []
        objects = np.random.random_integers(0, self.SELECTION_POOL - 1, num_obj)
        obj_queue_order = 0
        for object_idx in objects:
            # If mesh files and folders are stored in a different format, string parsings should be adapted throughout
            curr_mesh_file = os.path.join(self.obj_mesh_dir, "%01d.obj" % object_idx)
            # curr_mesh_file = os.path.join(self.obj_mesh_dir, "{:03d}/{:03d}.obj".format(object_idx, object_idx))
            curr_shape_name = 'iter%07d' % iteration_no + '_order%03d' % obj_queue_order + '_obj%03d' % object_idx
            obj_queue_order = obj_queue_order + 1

            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0]) * np.random.random_sample() + \
                     self.workspace_limits[0][0]
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0]) * np.random.random_sample() + \
                     self.workspace_limits[1][0]
            object_position = [drop_x, drop_y, self.DROP_HEIGHT]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]

            print('dropping  -->', curr_mesh_file)
            object_order.append(["%03d.obj" % object_idx, object_position, object_orientation])
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1],
                            self.obj_mesh_color[object_idx][2]]
            # 'importMesh' is also possible -- int float strings
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                                  'remoteApiCommandServer',
                                                                                                  vrep.sim_scripttype_childscript,
                                                                                                  'importShape',
                                                                                                  [0, 0, 255, 0],
                                                                                                  object_position + object_orientation + object_color,
                                                                                                  [curr_mesh_file,
                                                                                                   curr_shape_name],
                                                                                                  bytearray(),
                                                                                                  vrep.simx_opmode_blocking)

            if ret_resp == 8:
                print('Failed to add new objects to simulation.')
                time.sleep(0.5)
                return [-1, -1]
            curr_shape_handle = ret_ints[0]
            object_handles.append(curr_shape_handle)
            time.sleep(0.04)
        # wait a second after all objects were released
        time.sleep(0.3)

        return [object_handles, object_order]

    def _remove_objects(self, object_handles):

        ret_values = []
        for i in object_handles:
            ret = vrep.simxRemoveObject(self.sim_client, i, vrep.simx_opmode_blocking)
            ret_values.append(ret)
        if ret_values.count(0) != len(object_handles):
            print("WARNING: One or more items might not be removed!")

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
    config_file = os.getcwd() + '/database_generator/database_config.yaml'
    with open(config_file) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    dg = DatabaseGenerator(configuration)
    dg.generate_database()

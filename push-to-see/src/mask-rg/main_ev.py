from model import MaskRGNetwork
from dataset import PushAndGraspDataset
import yaml
import numpy as np
import torchvision.transforms as T
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from rewards import RewardGenerator
import os


def main():

    npy_path = '~/Database_vrep_inria_3/depth_ims/NUMPY/depth_000942.npy'
    npy_path = os.path.expanduser(npy_path)
    img_rgb_path = '~/Database_vrep_inria_3/color_ims/color_image_000942.png'
    img_rgb_path = os.path.expanduser(img_rgb_path)
    gt_path = '~/Database_vrep_inria_3/segmentation_masks/PNG/segmask_image_000942.png'
    gt_path = os.path.expanduser(gt_path)
    gt_info_path = '~/Database_vrep_inria_3/segmentation_masks/NUMPY/segmask_000942.npy'
    gt_info_path = os.path.expanduser(gt_info_path)
    pixel_label = np.load(gt_info_path)
    # print(pixel_label)

    with open('../../model_config.yaml') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    # create the model
    model = MaskRGNetwork(configuration)


    model.load_weights()
    # load a new dataset for the evaluation
    # model.set_data(dataset_test, is_test=True, img_id=3)

    # evaluate
    # res = model.evaluate_model()
    # np.save('results.npy', res)
    # with open('results.txt', 'w') as output:
    #     output.write(res)

    # depth_image = Image.open(img_path).convert("RGB")
    color_img = Image.open(img_rgb_path).convert("RGB")


    dep_npy = np.load(npy_path).astype(dtype=np.int16)
    # all vals are in between 0 to 250
    dep_npy = np.round(dep_npy / 20).astype(np.uint8)
    dep_npy = np.repeat(dep_npy.reshape(1024, 1024, 1), 3, axis=2)
    tt = T.ToTensor()
    depth_tensor = tt(dep_npy)
    if torch.cuda.is_available():
        depth_tensor = depth_tensor.cuda()

    testres = model.eval_single_img([depth_tensor])
    print(f'boxes predicted by maskrcnn: {np.shape(testres[0]["boxes"])[0]}')
    #
    # # read a GT mask and reformat
    gt = cv2.imread(gt_path)
    gt = np.asarray(gt)
    # print(np.shape(gt))
    gt = gt[:, :, :1]
    gt = np.squeeze(gt, axis=2)
    print(np.unique(gt))
    #
    rg = RewardGenerator(confidence_threshold=0.75, mask_threshold=0.5)
    rg.set_element(testres, gt, pixel_label)
    inf0 = rg.get_reward()

    # rg.print_alignedmask_gt()

    # rewards.test(testres,testres)
    print('num boxes --> ', len(testres[0]['boxes']))
    print('num masks --> ', len(testres[0]['masks']))

    #model.print_boxes(color_img, testres, score_threshold=0.75)
    # model.print_masks(color_img, testres, score_threshold=0.75)

    # modified, chosen best match mask for each gt obj
    rg.print_masks()

if __name__ == "__main__":
    main()

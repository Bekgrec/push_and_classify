from model import MaskRGNetwork
from dataset import PushAndGraspDataset
import yaml
import numpy as np
import os
from torch.utils import data as td
import torchvision.transforms as T
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from rewards import RewardGenerator

# img_path = '/home/baris/Database_vrep_inria/depth_ims/PNG/depth_image_012500.png'
# img_rgb_path = '/home/baris/Database_vrep_inria/color_ims/color_image_012500.png'
# gt_path = '/home/baris/Database_vrep_inria/segmentation_masks/PNG/segmask_image_012500.png'

def main():

    with open('../../model_config.yaml') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    # create the model
    model = MaskRGNetwork(configuration)

    # create dataset objects and set the model
    # dataset = PushAndGraspDataset(configuration)
    dataset = PushAndGraspDataset(configuration)
    test_indices = os.path.join(configuration['dataset']['path'], configuration['dataset']['test_indices'])
    test_subset = td.Subset(dataset, test_indices)
    # Training:
    # model.set_data(dataset)
    # model.train_model()


    # Evaluation:
    # this loads the saved weights from the file in the config file
    model.load_weights()
    # load a new dataset for the evaluation
    model.set_data(test_subset, is_test=True, batch_size=20)

    # evaluate
    res = model.evaluate_model()
    np.save('results.npy', res)
    with open('results.txt', 'w') as output:
        output.write(res)

    # depth_image = Image.open(img_path).convert("RGB")
    # color_img = Image.open(img_rgb_path).convert("RGB")
    # tt = T.ToTensor()
    # depth_tensor = tt(depth_image)
    # depth_tensor = depth_tensor.cuda()
    # testres = model.eval_single_img([depth_tensor])
    #
    # # read a GT mask and reformat
    # gt = cv2.imread(gt_path)
    # gt = np.asarray(gt)
    # gt = gt[:, :, :1]
    # gt = np.squeeze(gt, axis=2)
    #
    # rg = RewardGenerator(confidence_threshold=0.75, mask_threshold=0.75)
    # rg.set_element(testres, gt)
    # print(rg.get_reward())

    # rewards.test(testres,testres)
    # print('num boxes --> ', len(testres[0]['boxes']))
    # print('num masks --> ', len(testres[0]['masks']))

    # model.print_masks(color_img, testres, score_threshold=0.75)
    # print(testres)




if __name__ == "__main__":
    main()

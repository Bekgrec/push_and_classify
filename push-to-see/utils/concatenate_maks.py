import numpy as np
import cv2
import os
import glob

# PATH = '/home/baris/Workspace/push-to-see/logs/2021-03-10.01:20:29/mask-rg/prediction-images'
PATH = '/home/baris/Workspace/push-to-see/logs/2021-03-10.09:55:46/mask-rg/prediction-images'
PATH_TO_SAVE = '/home/baris/Desktop/sess_imgs'

file_list = sorted(glob.glob(PATH+'/*.mask_pred_diff.png'))


black_img = np.zeros([1024, 1024, 3]).astype(np.uint8)
white_img = np.ones([1024, 1024, 3]).astype(np.uint8) * 38
j = 1 # for file name index

for sel_file in file_list:
    head, tail = os.path.split(sel_file)
    split_file = tail.split('.')
    iteration_no = split_file[0]
    trial_no = split_file[1]
    if int(trial_no) == 0:

        sess_start_it = int(iteration_no)
        images = []
        for i in range(0, 11):
            if os.path.isfile(os.path.join(PATH, '{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it+i, i))):
                print('{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it+i, i))
                images.append(cv2.imread(os.path.join(PATH, '{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it + i, i))))
            else:
                images.append(white_img)

        img1 = cv2.hconcat(images)
        images = [black_img]

        for i in range(11, 21):
            if os.path.isfile(os.path.join(PATH, '{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it+i, i))):
                print('{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it + i, i))
                images.append(cv2.imread(os.path.join(PATH, '{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it + i, i))))
            else:
                images.append(white_img)

        img2 = cv2.hconcat(images)
        img = cv2.vconcat([img1, img2])
        images = [black_img]

        for i in range(21, 31):
            if os.path.isfile(os.path.join(PATH, '{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it+i, i))):
                print('{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it + i, i))
                images.append(cv2.imread(os.path.join(PATH, '{:06d}.{:02d}.mask_pred_diff.png'.format(sess_start_it + i, i))))
            else:
                print(i)
                images.append(white_img)

        img3 = cv2.hconcat(images)
        img = cv2.vconcat([img, img3])


        cv2.imwrite(os.path.join(PATH_TO_SAVE, "sess_{}_images_start_from_{}.jpg".format(j, sess_start_it)), img)
        j+=1


# img1 = cv2.imread(os.path.join(PATH, '000000.00.mask_pred_diff.png'))
# img2 = cv2.imread(os.path.join(PATH, '000001.01.mask_pred_diff.png'))
#
# img = cv2.hconcat([img1, img2])
# cv2.imwrite("/home/baris/img.png", img)
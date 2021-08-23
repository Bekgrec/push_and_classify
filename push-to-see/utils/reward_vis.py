import numpy as np
import matplotlib.pyplot as plt

color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                               [89.0, 161.0, 79.0],  # green
                               [156, 117, 95],  # brown
                               [242, 142, 43],  # orange
                               [237.0, 201.0, 72.0],  # yellow
                               [186, 176, 172],  # gray
                               [255.0, 87.0, 89.0],  # red
                               [176, 122, 161],  # purple
                               [118, 183, 178],  # cyan
                               [255, 157, 167]]) / 255.0  # pink
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-29.03:08:20/transitions/label-value.log.txt'
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-29.03:08:20-4k_keep/mask-rg/segmentation.txt'
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-29.03:08:20-4k_keep/transitions/label-value.log.txt' # training
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-29.16:07:48/transitions/predicted-value.log.txt' # test
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-30.04:04:47-keep-14k/transitions/loss-value.log.txt'
FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-30.04:04:47-keep-14k/mask-rg/segmentation.txt'

FILE_IT = '/home/baris/Workspace/push-to-see/logs/2021-01-30.04:04:47-keep-14k/session_success_fail_list.txt'

# FILE_IT= '/home/baris/Workspace/push-to-see/logs/2021-06-01.20:47:53/session_success_fail_list.txt'

f = open(FILE, "r")
asd = []

for line in f:
    qwe = line.replace(" \n", "")
    qwe = qwe.replace("[", "")
    qwe = qwe.replace("]", "")
    zxc = qwe.split(',')
    asd.append(float(zxc[3]))
    # asd.append(float(line))

f2 = open(FILE_IT, "r")
iter = []

for sess in f2:
    iteration = sess.split('tr_it --> ')
    iteration = iteration[1].split(')')
    iter.append(iteration[0])
test = []
first = 1
sessit = 0
col = []
for i in range(0, int(iter[-1])):
    test.append(sessit)
    col.append(color_space[sessit % 10])
    a = int(iter[sessit])
    if first == int(iter[sessit]):
        sessit+=1
    first += 1

print(len(iter))
print(len(asd))
# plt.bar(test,asd[:a], color = col)
plt.plot(asd)
plt.show()
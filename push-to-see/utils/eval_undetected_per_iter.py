import numpy as np
import matplotlib.pyplot as plt

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-03-10.01:20:29/mask-rg/err_rate_per_iteration.txt'

# pseudo-random
FILE = '/home/baris/Desktop/pseudo_random.txt'

FILE = '/home/baris/Workspace/push-to-see/logs/2021-06-01.20:47:53/mask-rg/err_rate_per_iteration.txt' # training first 19k
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-06-05.14:16:41/mask-rg/err_rate_per_iteration.txt' # training second 22k

# FILE = '/home/baris/Desktop/err_rate_per_iteration_41k.txt'

FILE = '/logs/2021-06-09.15:56:18/mask-rg/err_rate_per_iteration.txt'
FILE = '/logs/2021-06-09.22:19:39/mask-rg/err_rate_per_iteration.txt'

all_data = []
sess_data = []
temp_sess = []

# read and split --> [#objects, #undetected, #undetected/#objects, iteration_no, session_iteration_no]
f = open(FILE, 'r')
for line in f:
    asd = line.split(',')
    data = [int(asd[0].split('[')[1]), int(asd[1]), float(asd[2]), int(asd[3].split('[')[1]), int(asd[4].split(']')[0])]
    all_data.append(data)

i = 0
next_sess = True



while i+1 < len(all_data)-1:
    while next_sess:
        temp_sess.append(all_data[i])
        i += 1
        if i + 1 == len(all_data)-1:
            break
        if all_data[i+1][4] == 0:
            next_sess = False
    # add the last element of the session
    temp_sess.append(all_data[i])
    i += 1

    sess_data.append(temp_sess)
    temp_sess = []
    next_sess = True

sess_avg = np.zeros([31])
for session in sess_data:
    sess = np.asarray(session)
    for s_iter in range(0, sess.shape[0]):
        sess_avg[s_iter] = sess_avg[s_iter] + sess[s_iter][1]


sess_avg = sess_avg / len(sess_data)
print(sess_avg)
plt.bar(range(0, 31), sess_avg)
plt.show()
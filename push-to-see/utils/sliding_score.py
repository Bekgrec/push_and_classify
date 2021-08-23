import numpy as np
import matplotlib.pyplot as plt

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-16.13:55:09/session_success_fail_list.txt'

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-05.01:03:15/session_success_fail_list.txt'
FILE = '/home/baris/Workspace/push-to-see/logs/2021-03-10.01:20:29/session_success_fail_list.txt'

FILE = '/home/baris/Workspace/push-to-see/logs/2021-06-01.20:47:53/session_success_fail_list.txt'

FILE = '/home/baris/Workspace/push-to-see/logs/2021-06-05.14:16:41/session_success_fail_list.txt'
# FILE = '/media/baris/Data/last_logs_all_pushtosee/logs/2021-03-10.01:20:29/session_success_fail_list.txt'

# FILE = '/home/baris/Desktop/all_training.txt'
FILE = '/home/baris/Desktop/41k_all_1466_sess_succ.txt'
session = []
all_ses = []
all_print = []
all_iters= []
f = open(FILE, "r")
for line in f:
    asd = line.split('Session no ')
    asd = asd[1].split(' --> after ')
    # session.append(int(asd[0]))
    asd = asd[1].split(' --> ')
    iteration = asd[2].split(')')
    # all_iters.append(int(iteration[0]))
    action = asd[0].split(' ')
    session.append(int(action[0]))
    # asd = asd[2].split(')')
    # session.append(int(asd[0]))
    if int(session[0]) != 0: # ignore err sessions

        all_ses.append(session)
        all_print.append(int(session[0]))
    session = []

all_selected = np.asarray(all_ses)
unique, counts = np.unique(all_selected, return_counts=True)
total = all_selected.size

sliding_err = []
# for i in range(0, total - 20):
#     selected_window = all_selected[i: i + 20]
step = True
acc=0
j = 0
for i in range(0, total- 30):

    while step:
        acc += all_selected[i + j]
        j += 1
        if acc > 400:
            acc = 0
            selected_window = all_selected[i: i + j]
            j = 0
            step = False
    step = True

    unique_w, counts_w = np.unique(selected_window, return_counts=True)
    total_w = selected_window.size # this should be always 10

    if unique_w[counts_w.size - 1] == 31:
        total_valid = total_w
        total_fail = counts_w[counts_w.size - 1]

        total_succ = total_valid - total_fail
        # succ_sum = selected_window[selected_window !=[31]]
        # succ_sum = succ_sum[succ_sum!=[0]]
        # succ_sum = succ_sum.sum()

        sliding_err.append(((total_fail/total_valid)*100))

    else:
        print(unique_w)
        total_succ = total_w
        sliding_err.append(0)



# print(sliding_err)
# plt.plot(sliding_err)
plt.plot(np.asarray(sliding_err))
plt.show()
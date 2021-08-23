import numpy as np
import matplotlib.pyplot as plt

# Session no 1 --> after 31 pushing actions --> FAIL!  (tr_it --> 30)
# Session no 2 --> after 31 pushing actions --> FAIL!  (tr_it --> 61)
#  Session no 3 --> after 15 pushing actions --> SUCCESS!  (tr_it --> 77)
#  Session no 7 --> after 0 TRAINING IGNORED --> (tr_it --> 171)


# FILE = '/home/baris/Workspace/push-to-see/session_success_fail_list_8k.txt'
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-16.02:01:38-keep-2k/session_success_fail_list.txt'

######################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-25.12:55:52-test/session_success_fail_list.txt' # test model 0.9 - not many sessions
# % error --> 55.00000000000001 The average number of pushes before success:  17.88888888888889
# total --> 24 , total valid --> 20 , total fail --> 11

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-25.14:53:15-test/session_success_fail_list.txt' # test model 0.8
# % error --> 4.081632653061225 The average number of pushes before success:  8.574468085106384
# total --> 76 , total valid --> 49 , total fail --> 2

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-26.02:07:16/session_success_fail_list.txt' # random 0.85 conf and mask thresholds 0.75
# % error --> 27.27272727272727 The average number of pushes before success:  11.160714285714286
# total --> 310 , total valid --> 231 , total fail --> 63

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-25.20:33:57/session_success_fail_list.txt' # random 0.8 conf and mask thresholds 0.75
# % error --> 21.21212121212121 The average number of pushes before success:  12.384615384615385
# total --> 104 , total valid --> 66 , total fail --> 14

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-28.13:02:14/session_success_fail_list.txt' # test model 0.85 conf and mask thresholds 0.75
# % error --> 11.627906976744185 The average number of pushes before success:  10.631578947368421
# total --> 56 , total valid --> 43 , total fail --> 5

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-28.15:24:45/session_success_fail_list.txt' # random model 0.85 - conf and mask thresholds 0.8
# % error --> 75.0 The average number of pushes before success:  15.25
# total --> 18 , total valid --> 16 , total fail --> 12

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-28.17:16:23-keep/session_success_fail_list.txt' # test trained (8k tr w diff thresholds) model 0.85 - conf and mask thresholds 0.8
# % error --> 45.614035087719294 The average number of pushes before success:  11.290322580645162
# total --> 61 , total valid --> 57 , total fail --> 26

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-29.03:08:20-4k_keep/session_success_fail_list.txt' # new training 4k with  0.85 - conf 0.8  mask 0.8

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-29.16:07:48_keep_test/session_success_fail_list.txt' # test -new training with the same param
# % error --> 25.0 The average number of pushes before success:  14.083333333333334
# total --> 34 , total valid --> 32 , total fail --> 8

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-01-30.04:04:47-keep-14k/session_success_fail_list.txt' # training 14k - 0.85 - detection thresholds (0.8/0.8)

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-01.15:18:47/session_success_fail_list.txt' # model-14k -> 0.85 - detection thresholds (0.8/0.8) num obj --> 26-32
# new loc
# FILE = '/media/baris/Data/LOGS_SONDAN_DA_SON/2021-02-01.15:18:47/session_success_fail_list.txt'
# No ignored session! all sessions are valid! total --> 20 , total valid --> 20 , total fail --> 8
# % error --> 40.0 The average number of pushes before success:  17.666666666666668
# -----------------------------------------------------------
# LESS THAN 500 iterations for testing!
# No ignored session! all sessions are valid!
# total --> 20 , total valid --> 20 , total fail --> 8
# % error --> 40.0
# The average number of pushes before success:  17.666666666666668


# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-01.17:29:10/session_success_fail_list.txt' # random model -> 0.85 - detection thresholds (0.8/0.8) num obj --> 26-32
# No ignored session! all sessions are valid! total --> 48 , total valid --> 48 , total fail --> 40
# % error --> 83.33333333333334 The average number of pushes before success:  23.875

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-02.01:35:24/session_success_fail_list.txt' # model-14k -> 0.85 - detection thresholds (0.8/0.8) num obj --> 18 - 24
# FILE = '/media/baris/Data/LOGS_SONDAN_DA_SON/2021-02-02.01:35:24/session_success_fail_list.txt'
# total --> 181 , total valid --> 168 , total fail --> 36
# % error --> 21.428571428571427 The average number of pushes before success:  12.227272727272727
# -------------------------------------------------------------------------------------------------
# total --> 44 , total valid --> 40 , total fail --> 9
# % error --> 22.5
# The average number of pushes before success:  13.806451612903226



# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-04.18:16:22-keep-pseudo-rand/session_success_fail_list.txt' # pseudo-random --> 0.85 - detection thresholds (0.8/0.8) num obj --> 26-32
# No ignored session! all sessions are valid!
# total --> 26 , total valid --> 26 , total fail --> 11
# % error --> 42.30769230769231 The average number of pushes before success:  16.866666666666667

# FILE ='/home/baris/Workspace/push-to-see/logs/2021-02-04.21:05:22/session_success_fail_list.txt' # pseudo-random --> 0.85 - detection thresholds (0.8/0.8) num obj --> 18 - 24
# total --> 54 , total valid --> 49 , total fail --> 7
# % error --> 14.285714285714285 The average number of pushes before success:  12.785714285714286



# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-05.01:03:15/session_success_fail_list.txt' # Second time to get more samples --> pseudo-random --> 0.85 - detection thresholds (0.8/0.8) num obj --> 26-32
# total --> 98 , total valid --> 97 , total fail --> 50
# % error --> 51.546391752577314 The average number of pushes before success:  18.02127659574468


# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-05.11:38:40/session_success_fail_list.txt' # Second time to get more samples --> model-14k -> 0.85 - detection thresholds (0.8/0.8) num obj --> 26-32
# No ignored session! all sessions are valid!
# total --> 49 , total valid --> 49 , total fail --> 27
# % error --> 55.10204081632652 The average number of pushes before success:  17.363636363636363


# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-12.14:13:24/session_success_fail_list.txt' # 13k
# No ignored session! all sessions are valid!
# total --> 44 , total valid --> 44 , total fail --> 30
# % error --> 68.18181818181817 The average number of pushes before success:  17.714285714285715

# 13k+5k - test - tr problem
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-15.22:18:01/session_success_fail_list.txt'

# 2.2k training
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-16.02:43:45/session_success_fail_list.txt'

# 2.2k test - only 19 sess - 534it
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-16.11:42:17/session_success_fail_list.txt'
# total --> 19 , total valid --> 18 , total fail --> 15
# % error --> 83.33333333333334
# The average number of pushes before success:  22.0


######################################################################################################################
######################################################################################################################

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-17.22:31:57/session_success_fail_list.txt' # 10k -- [26 - 32]
# total --> 136 , total valid --> 135 , total fail --> 40
# % error --> 29.629629629629626  The average number of pushes before success:  15.83157894736842
# -----------------------------------------------------------
# No ignored session! all sessions are valid!
# total --> 36 , total valid --> 36 , total fail --> 8
# % error --> 22.22222222222222
# The average number of pushes before success:  16.5

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-18.10:00:29/session_success_fail_list.txt' # 10k -- [18 - 24]
#total --> 222 , total valid --> 205 , total fail --> 20
# % error --> 9.75609756097561 The average number of pushes before success:  8.8
# -----------------------------------------------------------
# total --> 61 , total valid --> 57 , total fail --> 6
# % error --> 10.526315789473683
# The average number of pushes before success:  9.72549019607843

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-18.17:50:30/session_success_fail_list.txt' # 5k -- [18 - 24]
# total --> 54 , total valid --> 52 , total fail --> 12
# % error --> 23.076923076923077 The average number of pushes before success:  12.875
# -----------------------------------------------------------
# total --> 43 , total valid --> 42 , total fail --> 10
# % error --> 23.809523809523807
# The average number of pushes before success:  12.65625

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-18.20:46:56/session_success_fail_list.txt' # 5k -- [26 - 32]
# No ignored session! all sessions are valid!
# total --> 46 , total valid --> 46 , total fail --> 27
# % error --> 58.69565217391305  The average number of pushes before success:  19.263157894736842
# -----------------------------------------------------------
# No ignored session! all sessions are valid!
# total --> 27 , total valid --> 27 , total fail --> 17
# % error --> 62.96296296296296
# The average number of pushes before success:  18.8

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-19.01:39:06/session_success_fail_list.txt' # 7.5k -- [26 - 32]
# total --> 108 , total valid --> 107 , total fail --> 32
# % error --> 29.906542056074763 The average number of pushes before success:  15.373333333333333
# -----------------------------------------------------------
# No ignored session! all sessions are valid!
# total --> 33 , total valid --> 33 , total fail --> 11
# % error --> 33.33333333333333
# The average number of pushes before success:  17.40909090909091


######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-19.11:10:25/session_success_fail_list.txt' # 7.5k -- [18 - 24]
# total --> 52 , total valid --> 48 , total fail --> 13
# % error --> 27.083333333333332 The average number of pushes before success:  13.142857142857142
# -----------------------------------------------------------
# total --> 40 , total valid --> 37 , total fail --> 11
# % error --> 29.72972972972973
# The average number of pushes before success:  14.5

######################################################################################################################
#### todo only 750 it!!
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-19.14:04:11/session_success_fail_list.txt' # 6k -- [18 - 24]
# total --> 54 , total valid --> 46 , total fail --> 10
# % error --> 21.73913043478261 The average number of pushes before success:  10.86111111111111
# -----------------------------------------------------------
# total --> 54 , total valid --> 46 , total fail --> 10
# % error --> 21.73913043478261
# The average number of pushes before success:  10.86111111111111

#####################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-19.16:26:22/session_success_fail_list.txt' # 6k -- [26 - 32]
# No ignored session! all sessions are valid!
# total --> 66 , total valid --> 66 , total fail --> 34
# % error --> 51.515151515151516 The average number of pushes before success:  16.78125
# -----------------------------------------------------------
# No ignored session! all sessions are valid!
# total --> 28 , total valid --> 28 , total fail --> 17
# % error --> 60.71428571428571
# The average number of pushes before success:  18.818181818181817

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-19.22:58:40/session_success_fail_list.txt' # 4k [18 - 24]
# total --> 110 , total valid --> 99 , total fail --> 11
# % error --> 11.11111111111111
# The average number of pushes before success:  9.647727272727273
# -----------------------------------------------------------
# total --> 60 , total valid --> 55 , total fail --> 5
# % error --> 9.090909090909092
# The average number of pushes before success:  10.72

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-20.03:00:31/session_success_fail_list.txt' # 4k [26 - 32]
# No ignored session! all sessions are valid!
# total --> 135 , total valid --> 135 , total fail --> 55
# % error --> 40.74074074074074
# The average number of pushes before success:  16.2375
# -----------------------------------------------------------
# No ignored session! all sessions are valid!
# total --> 31 , total valid --> 31 , total fail --> 15
# % error --> 48.38709677419355
# The average number of pushes before success:  15.9375

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-20.15:16:24/session_success_fail_list.txt' # 3k [18 - 24]
# total --> 46 , total valid --> 41 , total fail --> 14
# % error --> 34.146341463414636
# The average number of pushes before success:  11.296296296296296
# -----------------------------------------------------------
# total --> 44 , total valid --> 39 , total fail --> 13
# % error --> 33.33333333333333
# The average number of pushes before success:  11.5

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-20.17:38:44/session_success_fail_list.txt' # 3k [26 - 32]
# total --> 213 , total valid --> 210 , total fail --> 93
# % error --> 44.285714285714285 The average number of pushes before success:  16.641025641025642
# -----------------------------------------------------------
# No ignored session! all sessions are valid!
# total --> 30 , total valid --> 30 , total fail --> 13
# % error --> 43.333333333333336
# The average number of pushes before success:  18.352941176470587


######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-21.13:53:12/session_success_fail_list.txt' # 1k [18 - 24]
# total --> 106 , total valid --> 96 , total fail --> 20
# % error --> 20.833333333333336 The average number of pushes before success:  12.447368421052632
# -----------------------------------------------------------
# total --> 51 , total valid --> 45 , total fail --> 8
# % error --> 17.77777777777778
# The average number of pushes before success:  11.945945945945946


######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-21.19:00:59/session_success_fail_list.txt' #1k [26 -32]
# total --> 65 , total valid --> 63 , total fail --> 46
# % error --> 73.01587301587301
# The average number of pushes before success:  19.764705882352942
# -----------------------------------------------------------
# total --> 28 , total valid --> 26 , total fail --> 21
# % error --> 80.76923076923077
# The average number of pushes before success:  15.8

######################################################################################################################
# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-22.02:11:09/session_success_fail_list.txt' #500it [26 - 32]
# No ignored session! all sessions are valid!
# total --> 75 , total valid --> 75 , total fail --> 52
# % error --> 69.33333333333334
# The average number of pushes before success:  16.304347826086957
# -----------------------------------------------------------
# No ignored session! all sessions are valid!
# total --> 26 , total valid --> 26 , total fail --> 21
# % error --> 80.76923076923077
# The average number of pushes before success:  13.0

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-22.10:15:37/session_success_fail_list.txt' # 500it [18 - 24]
# < 750
# total --> 40 , total valid --> 37 , total fail --> 12
# % error --> 32.432432432432435
# The average number of pushes before success:  13.6

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-22.13:58:11/session_success_fail_list.txt' # 2k [26 - 32]
# < 750 No ignored session! all sessions are valid!
# total --> 27 , total valid --> 27 , total fail --> 19
# % error --> 70.37037037037037
# The average number of pushes before success:  16.625

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-22.22:24:43/session_success_fail_list.txt' # 9k [18 -24]
# total --> 53 , total valid --> 49 , total fail --> 8
# % error --> 16.3265306122449
# The average number of pushes before success:  10.658536585365853


# FILE = '/home/baris/Workspace/push-to-see/logs/2021-02-23.01:33:51/session_success_fail_list.txt' #9k [26 - 32]
# No ignored session! all sessions are valid!
# total --> 32 , total valid --> 32 , total fail --> 11
# % error --> 34.375
# The average number of pushes before success:  18.047619047619047

# FILE = '/home/baris/Workspace/push-to-see/logs/2021-03-10.01:20:29/session_success_fail_list.txt'

FILE = '/home/baris/Workspace/push-to-see/logs/2021-06-01.20:47:53/session_success_fail_list.txt'
###########################
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
    all_iters.append(int(iteration[0]))
    action = asd[0].split(' ')
    session.append(int(action[0]))
    # asd = asd[2].split(')')
    # session.append(int(asd[0]))
    all_ses.append(session)
    all_print.append(int(session[0]))
    session = []

# print(all_iters)
# consider solely the results in the first 750 iteration
all_iters = np.asarray(all_iters)
print(all_iters[-1], '--> < 2000')
if all_iters[-1] > 1999:
    all_it_valid = np.where(all_iters < 2000)
    all_selected = np.asarray(all_ses[:all_it_valid[0].size])
else:
    all_selected = np.asarray([0])

unique, counts = np.unique(all_selected, return_counts=True)
total = all_selected.size

if unique[0] == 0 and unique[counts.size-1] == 31: # !!!add no error cases
    total_valid = total - counts[0]
    total_fail = counts[counts.size-1]
    print('total --> %d , total valid --> %d , total fail --> %d ' % (total, total_valid, total_fail))
    print('% error -->', (total_fail/total_valid)*100)
    total_succ = total_valid - total_fail
    succ_sum = all_selected[all_selected !=[31]]
    succ_sum = succ_sum[succ_sum!=[0]]
    succ_sum = succ_sum.sum()
    print('The average number of pushes before success: ', succ_sum/total_succ)
elif unique[0] != 0 and unique[counts.size-1] == 31:
    print('No ignored session! all sessions are valid!')
    total_valid = total
    total_fail = counts[counts.size-1]
    print('total --> %d , total valid --> %d , total fail --> %d ' % (total, total_valid, total_fail))
    print('% error -->', (total_fail/total_valid)*100)
    total_succ = total_valid - total_fail
    succ_sum = all_selected[all_selected !=[31]]
    succ_sum = succ_sum.sum()

    print('The average number of pushes before success: ', succ_sum/total_succ)
else:
    print("ERROR Check evaluation! no fail or no immediate success cases in the results.")

all_print = np.asarray(all_print)
plt.bar(list(range(0, all_print.size)), all_print)
plt.show()
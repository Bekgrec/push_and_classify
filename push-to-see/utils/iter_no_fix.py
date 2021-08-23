import numpy as np

FILE1 = '/home/baris/Desktop/first_part.txt'
FILE2 = '/home/baris/Desktop/second_part.txt'

all_data1 = []
all_data2 = []


# read and split --> [#objects, #undetected, #undetected/#objects, iteration_no, session_iteration_no]
f = open(FILE1, 'r')
for line in f:
    asd = line.split(',')
    data = [int(asd[0].split('[')[1]), int(asd[1]), float(asd[2]), int(asd[3].split('[')[1]), int(asd[4].split(']')[0])]
    all_data1.append(data)

f = open(FILE2, 'r')
for line in f:
    asd = line.split(',')
    data = [int(asd[0].split('[')[1]), int(asd[1]), float(asd[2]), int(asd[3].split('[')[1]), int(asd[4].split(']')[0])]
    all_data2.append(data)


all1 = np.asarray(all_data1)
# clean the sessions afters 19k
all1 = all1[:19012]

all2 = np.asarray(all_data2)
all2 = all2[:22006]
all2[:, 3] = all2[:,3] + 19012

results = np.concatenate((all1, all2),axis=0)
np.save('/home/baris/Desktop/np_41k.npy', results)
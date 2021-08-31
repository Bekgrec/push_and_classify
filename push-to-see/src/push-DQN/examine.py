import numpy as np
import matplotlib.pyplot as plt
import sys
import os

cur = os.getcwd()
print(cur)
gt = plt.imread(cur + '/ground_truth.png')
ngt = np.asarray(gt)
print(np.unique(ngt))

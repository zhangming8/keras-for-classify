import random

label = [77, 72, 88, 105, 61, 104, 90, 67]

# random.sample(label , 2)
#
# import numpy as np
# per = np.random.permutation()

for i in range(25):
    index1 = random.randint(0, len(label)-1)
    a = label[index1]

    index2 = random.randint(0, len(label)-1)
    while index1 == index2:
        index2 = random.randint(0, len(label) - 1)
    b = label[index2]

    print a, b
    # random.shuffle(label)

import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def merge(l):
    res = l[0]
    for i in range(1,len(l)):
        batch = l[i]
        res['data'] = np.concatenate([res['data'], batch['data']], axis=0)
        res['labels'] += batch['labels']
    return res

u = [unpickle('cifar-10-batches-py/data_batch_'+str(i+1)) for i in range(5)]
u = merge(u)


print u.keys()
print u['data'].shape
i=-1
c=-1

while c < 100-1:
    i+=1
    if u['labels'][i] != 1:
        continue
    else:
        c+=1
        print c
        img = u['data'][i]
        img = img.reshape([3,32,32])
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        plt.subplot(10,10,c+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        #plt.title(u['labels'][i])
plt.show()

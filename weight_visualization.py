# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_CNN([(1, 6, 5), (2,), (6, 16, 5), (2,),("reshape"),(16*5*5, 120),(120, 84), (84, 10)], "ReLU", [1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
model.load_model(r'.\saved_models\base_model.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

mats = []
mats.append(model.layers[0].params['kernel'])
mats.append(model.layers[3].params['kernel'])

# import ipdb; ipdb.set_trace()
_, axes = plt.subplots(3, 6)
_.set_tight_layout(1)
axes = axes.reshape(-1)
for i in range(6):
        axes[i].matshow(mats[0][i][0].reshape(5,5))
        axes[i].set_xticks([])
        axes[i].set_yticks([])
for j in range(6, 18):
        axes[j].matshow(mats[1][j-6][0].reshape(5,5))
        axes[j].set_xticks([])
        axes[j].set_yticks([])

# plt.figure()
# plt.matshow(mats[0][1][0])
# plt.xticks([])
# plt.yticks([])
plt.show()
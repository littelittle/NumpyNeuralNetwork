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

def kernel_visualization():
        mats = []
        mats.append(model.layers[0].params['kernel'])
        mats.append(model.layers[3].params['kernel'])

        # import ipdb; ipdb.set_trace()
        fig, axes = plt.subplots(2, 1, figsize=(8, 6)) # 创建一个包含两个子图的 figure

        # 可视化第一层卷积核
        axes[0].set_title('the first layer kernel(#6)')
        grid_first = axes[0].inset_axes([0, 0, 1, 1]) # 创建一个子区域用于绘制卷积核网格
        for i in range(6):
                ax = grid_first.inset_axes([i/6, 0, 1/6, 1]) # 在子区域中创建更小的 axes
                ax.matshow(mats[0][i][0].reshape(5, 5), cmap='viridis') # 使用 'viridis' 色图
                ax.set_xticks([])
                ax.set_yticks([])
        grid_first.set_xticks([]) # 隐藏 x 轴刻度
        grid_first.set_yticks([]) # 隐藏 y 轴刻度

        # 可视化第二层卷积核
        axes[1].set_title('the second layer kernel(#16)')
        grid_second = axes[1].inset_axes([0, 0, 1, 1]) # 创建一个子区域用于绘制卷积核网格
        for j in range(16):
                row = j//4
                col = j%4
                x_pos = col/4
                y_pos = row/4
                width = 1/4
                height = 1/4
                ax = grid_second.inset_axes([x_pos, y_pos, width, height]) # 在子区域中创建更小的 axes
                ax.matshow(mats[1][j][0].reshape(5, 5), cmap='viridis') # 使用 'viridis' 色图
                ax.set_xticks([])
                ax.set_yticks([])

        grid_second.set_xticks([]) # 隐藏 x 轴刻度
        grid_second.set_yticks([]) # 隐藏 y 轴刻度


        # fig.tight_layout()
        plt.show()

def feature_visualization(idx=0):
        """
        visualize the feature map of the first layer and the second layer
        """
        img = test_imgs[idx].reshape(1, 1, 28, 28)
        feature_map = model.layers[0].forward(img)
        feature_map = feature_map.reshape(6, 26, 26)
        fig, axes = plt.subplots(2, 3, figsize=(8, 6)) # 创建一个包含两个子图的 figure
        axes = axes.reshape(-1) # 将 axes 展平为一维数组
        fig.suptitle('the feature map of the first layer(#6)')
        for i in range(6):
                axes[i].imshow(feature_map[i], cmap='viridis')
                axes[i].set_title(f"feature map {i+1}")
                axes[i].axis('off') # 隐藏坐标轴
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
        # kernel_visualization()
        feature_visualization(1)
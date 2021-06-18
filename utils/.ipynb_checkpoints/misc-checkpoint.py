import numpy as np
import matplotlib.pyplot as plt


def show_image(img):
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)), interpolation="nearest")

    
def show_image_grid(dataset, num_samples=10):
    


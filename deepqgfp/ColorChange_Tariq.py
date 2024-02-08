import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    path = "7.jpg"
    i1 = cv2.imread(path)
    i2 = i1.copy()
    print(i2.shape)
    i2[...,1] = i1[...,2]
    i2[...,2] = i1[...,1]
    i2[...,0] = i1[...,1]
    print((i1==i2).all())
    plt.imshow(i2)
    plt.show()
    cv2.imwrite(path.replace('.png', '_changed.png'), i2)
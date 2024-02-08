import os
import numpy as np
import random
from skimage import io, color, transform
from functools import reduce
import matplotlib.pyplot as plt

def readImage(dir, file):
    return io.imread(os.path.join(dir, file))

def preprocess(image):
    return (image/255).flatten()

def imagesAndLabels(dir, label):
    images = []
    labels = []
    for filename in os.listdir(dir):
        if filename.endswith(".jpg"):
            image = readImage(dir, filename)
            images.append(preprocess(image))
            labels.append(label)
    return np.array(images), np.array(labels)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def monkeyOrNot(x):
    if x==1: return "Human"
    if x==0: return "Monkey"

images0, labels0 = imagesAndLabels("./dataset/test/monke", 0)
images1, labels1 = imagesAndLabels("./dataset/test/human", 1)

loaded_weights = np.load('weights.npz')
A_loaded, B_loaded = loaded_weights['A'], loaded_weights['B']

Y_regression_monkey = sigmoid(images0.dot(A_loaded) + B_loaded)
Y_regression_human = sigmoid(images1.dot(A_loaded) + B_loaded)
classification_monkey = list(map(lambda x: 1 if x >= 0.5 else 0, Y_regression_monkey))
classification_human = list(map(lambda x: 1 if x >= 0.5 else 0, Y_regression_human))

plt.figure(figsize=(15, 5))
for i in range(3, 8):
    plt.subplot(2, 5, i-3 + 1)
    plt.imshow(np.reshape(images0[i], (64, 64)), cmap='gray')
    plt.title(monkeyOrNot(classification_monkey[i]))
    plt.axis('off')

for i in range(3, 8):
    plt.subplot(2, 5, i-3 + 6)
    plt.imshow(np.reshape(images1[i], (64, 64)), cmap='gray')
    plt.title(monkeyOrNot(classification_human[i]))
    plt.axis('off')
plt.show()


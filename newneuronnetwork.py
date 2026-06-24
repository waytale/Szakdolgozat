import numpy as np

def check_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = np.fromfile(f, dtype='>i4', count=4)
        print(f"{path}:")
        print(" magic:", magic)
        print(" count:", num)
        print(" size:", rows, "x", cols)

def check_labels(path):
    with open(path, 'rb') as f:
        magic, num = np.fromfile(f, dtype='>i4', count=2)
        print(f"{path}:")
        print(" magic:", magic)
        print(" count:", num)

check_images("data/MNIST/raw/train-images-idx3-ubyte")
check_labels("data/MNIST/raw/train-labels-idx1-ubyte")
check_images("data/MNIST/raw/t10k-images-idx3-ubyte")
check_labels("data/MNIST/raw/t10k-labels-idx1-ubyte")



images = np.fromfile("data/MNIST/raw/train-images-idx3-ubyte", dtype=np.uint8)[16:]
print(images.min(), images.max())


from torchvision import datasets

mnist = datasets.MNIST(root="data", train=True, download=False)

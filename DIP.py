import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import matplotlib
plt.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] =False

def imread(path, mode='gray'):
    cvtMode = {
        'gray': cv2.COLOR_BGR2GRAY,
        'rgb': cv2.COLOR_BGR2RGB,
        'hsv': cv2.COLOR_BGR2HSV,
        'lab': cv2.COLOR_BGR2LAB,
        'yuv': cv2.COLOR_BGR2YUV,
        'ycrcb': cv2.COLOR_BGR2YCrCb,
    }
    if mode not in cvtMode:
        raise ValueError('mode should be one of {}'.format(cvtMode.keys()))
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cvtMode[mode])
    print(f'从{path}读取图像，大小为{img.shape}，模式为{mode}')
    return img


def imshow(img):
    plt.axis('off')
    plt.imshow(img, cmap=cm.gray, vmin=0, vmax=255)


def imwrite(path, img):
    cv2.imwrite(path, img)


def calcHist(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    return hist


def showHist(hist):
    plt.title('灰度直方图')
    plt.plot(hist)


def showImageWithHist(img, title='原图'):
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    imshow(img)
    plt.title(title)

    plt.subplot(122)
    hist = calcHist(img)
    showHist(hist)


def showImages(imgs,  titles=None, cols=3):
    n_imgs = len(imgs)
    rows = n_imgs // cols + 1
    plt.figure(figsize=(cols*4, rows*4))
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        if titles is not None:
            plt.title(titles[i])
        imshow(img)


def darkenImage(img, alpha = 10, multiple = False):
    if multiple:
        img_darkened = cv2.multiply(img, np.array([alpha]))
    else:
        img_darkened = cv2.subtract(img, np.array([alpha]))
    return img_darkened

# 直方图均衡化
def equalizeImage(img):
    img_equalized = cv2.equalizeHist(img)
    return img_equalized

def resizeImage(img, new_size):
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)


def showDiff(img1, img2, t1='图一', t2='图二', alpha=1):
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title(t1)
    imshow(img1)
    plt.subplot(122)
    plt.title(t2)
    imshow(img2)

    cmap = cm.gray
    fig = plt.figure(figsize=(4, 4))
    img_diff = img1.astype(np.int16) - img2.astype(np.int16)
    img_diff = img_diff * alpha

    plt.axis('off')
    plt.title('差值图')

    ax = plt.axes([0, 0.05, 0.9, 0.9])  # left, bottom, width, height

    im = ax.imshow(img_diff, interpolation='nearest', cmap=cmap)
    ax.axis(False)

    cax = plt.axes([0.95, 0.05, 0.05, 0.9])
    plt.colorbar(mappable=im, cax=cax)


def showImagesWithHist(imgs, titles=None):
    n_imgs = len(imgs)
    plt.figure(figsize=(8, n_imgs * 4))
    for i, img in enumerate(imgs):
        plt.subplot(n_imgs, 2, i * 2 + 1)
        imshow(img)

        if titles is not None:
            plt.title(titles[i])

        hist = calcHist(img)
        plt.subplot(n_imgs, 2, i * 2 + 2)
        showHist(hist)

        if titles is not None:
            plt.title(titles[i] + '-灰度直方图')



def calcFFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def showFFT(fshift):
    plt.imshow(np.log(1+np.abs(fshift)), cmap='gray')

def showImageWithFFT(img):
    fshift = calcFFT(img)
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.title('原图')
    imshow(img)
    plt.subplot(122)
    plt.title('频域图')
    showFFT(fshift)


def calcIFFT(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def showFFTWithImage(fshift):
    img_back = calcIFFT(fshift)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.title('频域图')
    showFFT(fshift)
    plt.subplot(122)
    plt.title('原图')
    imshow(img_back)


def frequencyFiltering(img, filter):
    fshift = calcFFT(img)
    f_filtered = fshift * filter
    img_back = calcIFFT(f_filtered)
    showFFTWithImage(f_filtered)
    return img_back


def gaussianFilter(h, w, sigma):
    # 获取索引矩阵及中心点坐标
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    # 计算中心距离矩阵
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # 计算变换矩阵
    filter = np.exp(- dis_square / (2 * sigma ** 2))

    return filter


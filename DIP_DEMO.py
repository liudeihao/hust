import cv2
from DIP import *

# Padding
def demo_padding(img):
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

    titles = ['ORIGINAL', 'REPLICATE', 'REFLECT', 'REFLECT_101', 'WRAP', 'CONSTANT']
    imgs = [img, replicate, reflect, reflect101, wrap, constant]

    showImages(imgs, titles)


def demo_equalize(img):
    img_darkened = darkenImage(img, alpha=0.05, multiple=True)
    img_equalized = equalizeImage(img_darkened)
    showImageWithHist(img_darkened, title='变暗的图像')
    showImageWithHist(img_equalized, title='直方图均衡化')


def demo_resize(img):
    img_64x64 = resizeImage(img, (64, 64))
    img_128x128 = resizeImage(img, (128, 128))

    showImages([img, img_64x64, img_128x128], titles=['原图', '64x64', '128x128'])


def demo_space_lp_filtering(img):
    # 空间域低通滤波
    # 均值滤波
    img_mean = cv2.blur(img, (5, 5))

    # 高斯滤波
    img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)

    # 中值滤波
    img_median = cv2.medianBlur(img, 5)

    # 双边滤波
    img_bilater = cv2.bilateralFilter(img, 9, 75, 75)

    # 展示不同的图片
    titles = ['srcImage', 'mean', 'Gaussian', 'median', 'bilateral']
    imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

    showImages(imgs, titles)


def demo_space_hp_filtering(img):
    # 空间域高通滤波
    # 拉普拉斯滤波
    laplacian_kernal = cv2.Laplacian(img, cv2.CV_64F)
    img_laplacian = cv2.convertScaleAbs(laplacian_kernal)  # 将负值转换为正值

    # Sobel滤波
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x方向
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    img_sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Prewitt滤波
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    img_prewitt_x = cv2.filter2D(img, -1, kernel_x)
    img_prewitt_y = cv2.filter2D(img, -1, kernel_y)
    img_prewitt = cv2.addWeighted(img_prewitt_x, 0.5, img_prewitt_y, 0.5, 0)

    titles = ['srcImage', 'laplacian', 'sobel', 'prewitt']
    imgs = [img, img_laplacian, img_sobel, img_prewitt]
    showImages(imgs, titles)


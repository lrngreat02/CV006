import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

# image reading
img_ori = cv2.imread('lenna.jpg', 1)  # flag=1, RGB
img_gray = cv2.imread('lenna.jpg', 0)  # flag=0, gray
print(img_gray.shape)

# cv2 show an image
cv2.imshow('lenna\'s photo', img_ori)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

# matplotlib show an image
plt.figure(figsize=(4, 4))
plt.imshow(img_gray, cmap='gray')   # w/o cmap, imshow will be 'green'
plt.show()

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(img_ori)
plt.subplot(122)
plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
plt.show()
# image reading from OPENCV(BGR)
# image reading from Matplotlib(RGB)
# image reading from PILLOW(RGB)


# define a function to open image
def my_show(img, size=(2, 2)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


my_show(img_ori)

my_show(img_ori[100:300, 200:300])

# channel split
B, G, R = cv2.split(img_ori)
print(B.shape)
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

# color channel change
def img_cooler(img, b_inc, r_dec):  # inc: increase, dec: decrease
    B, G, R = cv2.split(img)
    b_lim = 255 - b_inc
    B[B>b_lim] = 255
    B[B<=b_lim] = (B[B<=b_lim] + b_inc).astype(img.dtype)
    r_lim = r_dec
    R[R<r_lim] = 0
    R[R>=r_lim] = (R[R>=r_lim] - r_dec).astype(img.dtype)
    return cv2.merge((B, G, R))

xxx = img_cooler(img_ori, 20, 40)
my_show(xxx)

# Gamma Change, Look Up Table
def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(img, table)

img_dark = cv2.imread('dark.jpg', 1)
my_show(img_dark, (6,6))
img_brighter = adjust_gamma(img_dark, 2)
my_show(img_brighter,size=(6,6))

# histogram
plt.subplot(121)
plt.hist(img_dark.flatten(), 256, [0,256], color='b')
plt.subplot(122)
plt.hist(img_brighter.flatten(), 256, [0,256], color='r')
plt.show()


# histogram equalization  # YUV 色彩空间的Y 进行直方图均衡，调亮图片
img_yuv = cv2.cvtColor(img_dark,cv2.COLOR_BGR2YUV) # BGR转YUV
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # 对Y通道单独进行调整
img_yuv2BGR = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR) # YUV转BGR
my_show(img_yuv2BGR,size=(8,8))

# hist comparison
plt.subplot(131)
plt.hist(img_dark.flatten(),256,[0,256],color='r')
plt.subplot(132)
plt.hist(img_brighter.flatten(),256,[0,256],color='b')
plt.subplot(133)
plt.hist(img_yuv2BGR.flatten(),256,[0,256],color='g')
plt.show()


## transform
# perspective transform 1
pts1 = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])  # 源点
pts2 = np.float32([[0, 0], [0, 300], [400, 100], [500, 500]])  # 目标点
M = cv2.getPerspectiveTransform(pts1, pts2)  ## 计算单应性矩阵
img_warp = cv2.warpPerspective(img_ori, M, (500, 500))  # 通过M进行图片转换
my_show(img_warp)

# rotation
img = img_ori
M1 = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1)  # center是（列，行）， angle, scale
img_rotate = cv2.warpAffine(img, M1, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

print(M1)

# scale+rotation+translation = similarity transform
M2 = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.5)
img_rotate = cv2.warpAffine(img, M2, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

print(M2)

# Affine Transform
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('affine lenna', dst)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()


# perspective transform 2
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp

M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])
cv2.imshow('lenna_warp', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

# 膨胀，腐蚀
img_writing = cv2.imread('lb.png',0)
plt.figure(figsize=(10,8))
plt.imshow(img_writing,cmap='gray')
plt.show()

erode_writing = cv2.erode(img_writing,None,iterations=1)
plt.figure(figsize=(10,8))
plt.imshow(erode_writing,cmap='gray')
plt.show()

dilate_writing = cv2.dilate(img_writing,None,iterations=1)
plt.figure(figsize=(10,8))
plt.imshow(dilate_writing,cmap='gray')
plt.show()






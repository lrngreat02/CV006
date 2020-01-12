import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

# image read-in
img_ori = cv2.imread('lenna.jpg', 1)  # flag=1, RGB
img_gray = cv2.imread('lenna.jpg', 0)  # flag=0, gray

# define a function to open image
def my_show(img, size=(2, 2)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# image crop
def image_crop(img,up,lw,lh,rh ): # you code here
    r, c, ch = img.shape
    upper_bound = int(r * up)
    lower_bound = int(r * lw)
    left_bound = int(c * lh)
    right_bound = int(c * rh)
    return img[upper_bound:lower_bound, left_bound:right_bound]

img_x = image_crop(img_ori, 0.2, 0.8, 0.0, 0.5)
plt.imshow(cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB))
plt.show()


# color shift
def color_shift(img, b, g, r):
    B, G, R = cv2.split(img)
    for bgr,X in [[b,B], [g,G], [r,R]]:
        if bgr > 0:
            bgr_limit = bgr
            X[X>bgr_limit] = 255
            X[X<bgr_limit] = (bgr + X[X<bgr_limit]).astype(img.dtype)
        elif bgr < 0:
            bgr_limit = abs(bgr)
            X[X<bgr_limit] = 0
            X[X>bgr_limit] = (bgr + X[X>bgr_limit]).astype(img.dtype)
    return cv2.merge((B,G,R))

img_x2 = color_shift(img_ori, 50, -50, 30)
plt.imshow(cv2.cvtColor(img_x2, cv2.COLOR_BGR2RGB))
plt.show()

# image rotation transform
def rotation(img, center, angle, scale,):
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return img_rotate

img_rotate = rotation(img_ori, (400, 100), -15, 0.2)
cv2.imshow('rotation', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

# image perspective transform
def perspective_transform(img): # you code here
    height, width, channels = img.shape
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

M_warp, img_warp = perspective_transform(img_ori)
cv2.imshow('lenna_warp', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
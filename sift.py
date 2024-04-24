# 图像拼接
import time
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

MIN = 10    # 最小匹配点数，如果匹配的特征点数量大于最小数量，则进行图像拼接

# 读取两张图片，对第一张图片进行调整，使其与第二张图片大小一致,并展示两张图片
img1 = plt.imread("img_4/img2_ori.jpeg")
img2 = plt.imread("img_4/img1_ori.jpeg")

height2 = int(img2.shape[0])
width2 = int(img2.shape[1])
dim = (width2, height2)
img1 = cv.resize(img1, dim, interpolation=cv.INTER_AREA)

daipinjie_image = np.hstack((img1, img2))
# plt.imshow(daipinjie_image),plt.axis('off'), plt.show()
# plt.imsave("sift待拼接的两张图.png", daipinjie_image)

start_time = time.time()
# 使用SIFT检测关键点，并在原图像上绘制关键点，并打印特征点数目
sift = cv.xfeatures2d.SIFT_create(nOctaveLayers=4)

key1, describe1 = sift.detectAndCompute(img1, None)
key2, describe2 = sift.detectAndCompute(img2, None)

print(f"模板特征点数目：{len(key1)}")
print(f"待匹配图片特征点数：{len(key2)}")
tiqu_time = time.time()  #特征点提取时间

img3 = img1.copy() #复制img1并命名为img3，在img3上绘制关键点
img4 = img2.copy() #同上
img3 = cv.drawKeypoints(img3, key1, img3, ) #绘制关键点
img4 = cv.drawKeypoints(img4, key2, img4, )
tezhengtiqu_image = np.hstack((img3, img4))
# plt.imshow(tezhengtiqu_image),plt.axis('off'), plt.show()
# plt.imsave("sift特征提取.png", tezhengtiqu_image)

#使用FLANN算法进行特征点匹配，并打印总匹配对数量
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv.FlannBasedMatcher(indexParams, searchParams)
match = flann.knnMatch(describe1, describe2, k=2)
print(f"总匹配对数：{len(match)}")

# 仅保留距离最近的匹配特征点，以及距离最近的点与次近的点之间距离比例小于0.55的特征点，并画出匹配结果
good_matches = []
for i, (m, n) in enumerate(match):
    if m.distance < 0.3 * n.distance:
        good_matches.append(m)
print(f"正确匹配对数：{len(good_matches)}")
pipei_time = time.time() #特征点匹配时间

img_sift = cv.drawMatchesKnn(img1, key1, img2, key2, [good_matches], None, flags=2) 
# plt.imshow(img_sift),plt.axis('off'), plt.show()

## 如果匹配的特征点数量大于最小数量，则进行图像拼接
if len(good_matches) > MIN:
    src_pts = np.float32([key1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ano_pts = np.float32([key2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # 使用RANSAC算法进行图像校正
    M, mask = cv.findHomography(src_pts, ano_pts, cv.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = h1 + h2
    width = w1 + w2

    # 对第二张图片进行透视变换
    # warpImg = cv.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))
    warpImg = cv.warpPerspective(img2, np.linalg.inv(M), (width, height))

    # 将两张图片拼接在一起
    direct = warpImg.copy()
    # 将第一张图像粘贴到拼接后的图像的左上角
    direct[0:img1.shape[0], 0:img1.shape[1]] = img1
    # 将左右两张图片的重叠部分进行平滑处理，得到最终的拼接结果
    rows, cols = img1.shape[:2] # 获取第一张图片的行数和列数

    # 找到拼接后图像的左边界和右边界
    left = 0
    right = cols

    for col in range(0, cols):
        if img1[:, col].any() and warpImg[:, col].any():  # 开始重叠的最左端
            left = col
        break
    for col in range(cols - 1, 0, -1):
        if img1[:, col].any() and warpImg[:, col].any():  # 重叠的最右一列
            right = col
        break

    # 创建一张新的图像，用于存放拼接结果
    res = np.zeros([rows, cols, 3], np.uint8)

    # 对每个像素进行处理，将两张图像拼接起来
    for row in range(0, rows):
        for col in range(0, cols):
            if not img1[row, col].any():
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = img1[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

    # 将拼接后的图像放回warpImg中
    warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
    pinjie_time = time.time() #图像拼接总时间

    # 显示拼接前和拼接后的图像
    # plt.imshow(direct, ), plt.axis('off'), plt.show()  #直接拼接
    # plt.imshow(warpImg, ), plt.axis('off'), plt.show() #对边缘进行平滑处理
    
    # 输出拼接所用的时间
    print("特征点提取时间：%.2f" % (tiqu_time - start_time))
    print("特征点匹配时间：%.2f" % (pipei_time - tiqu_time))
    print("图像拼接总时间：%.2f" % (pinjie_time - pipei_time))
    print("总时间：%.2f" % (pinjie_time - start_time))

    # 保存拼接后的图像
    plt.imsave("sift原图.png", daipinjie_image)
    plt.imsave("sift特征提取.png", tezhengtiqu_image)
    plt.imsave("sift特征匹配.png", img_sift)
    plt.imsave("sift直接拼接.png", direct)
    plt.imsave("sift对边缘进行平滑处理.png", warpImg)

else:
    print("not enough matches!")


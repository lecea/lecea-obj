import tensorflow as tf
print ('hello, world!')

# # ---------- 容器（containers) --------------------
# ### (1) list （可变）列表，以方括号表示
# x = [0,1,2,3,4,5,6]        # 创建列表
# print(x[0], x[1], x[-1])   # 0, 1, 6    ## 序号从 0 开始, -1表示尾元素
#
# ## 切片操作 (slice)
# print(x[1:5:2])            # [1, 3]  ## 获取子序列 start:end:step
# print(x[:4:-1])            # [6, 5]  ## 获取子序列，这里的步长为为-负数，表示逆序
# print(x[:-3:-1])           # [6, 5]  ## 获取子序列，这里的步长为为-负数，表示逆序
# x[3:6] = [5,4,3]            #         ## 赋值(注意保证个数相同）
# print(x)                   # [0, 1, 2, 5, 4, 3, 6]
#
# ### (2) tuple （不可变）元组，以圆括号创建
# x = (0,1,2,3,4,5,6)    #创建元组
# print(x[0], x[1], x[-1])   # 0, 1, 6    ## 序号从 0 开始, -1表示尾元素
#
# ## 元组切片与列表切片类似，不过不可以修改
# # x[3] = -3 # TypeError: 'tuple' object does not support item assignment
#
# ### (3) dict (可变但索引不重复)字典，以花括号和冒号创建
# x = {"a":1, "b":2, "c":3}      ## Python 3.6 中字典跟有所变动
# print(x)                      # {'b': 2, 'a': 1, 'c': 3}
# print(x.keys())               # dict_keys(['b', 'a', 'c'])
# print(x.items())              # dict_items([('b', 2), ('a', 1), ('c', 3)])
# print(x["a"], x["b"], x["c"]) # 1 2 3    ## 按关键字索引
# x["b"] = "B"                   #           ## 修改
# print(x)      # {'b': 'B', 'a': 1, 'c': 3}
#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
#
# np.random.seed(0)
# data = np.random.randn(2, 100)
#
# fig, axs = plt.subplots(2, 2, figsize=(5, 5))
# ## 绘制子图
# axs[0, 0].hist(data[0])
# axs[1, 0].scatter(data[0], data[1])
# axs[0, 1].plot(data[0], data[1])
# axs[1, 1].hist2d(data[0], data[1])
#
#
# ## (2) 绘制图像
# img = mpimg.imread("004545.jpg")
# imgx = img[:,:,0] # 取第一个通道
#
#
# ## 创建画布
# fig = plt.figure()
#
# ## 绘制原始图像，并加上颜色条
# axs = fig.add_subplot(1,3,1)
# ipt = plt.imshow(img)
# axs.set_title("origin")
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
#
# ## 绘制伪彩色图像，并加上颜色条
# axs = fig.add_subplot(1,3,2)
# ipt = plt.imshow(imgx,cmap="winter")
# axs.set_title("winter")
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
#
# ## 绘制直方图
# axs = fig.add_subplot(1,3,3)
# ipt = plt.hist(imgx.ravel(), bins=256, range=(0, 1.0), fc='k', ec='k')
# axs.set_title("histogram")


## 显示
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('004545.jpg')
#
#
# # BGR=> Gray; 高斯滤波; Canny 边缘检测
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gaussed = cv2.GaussianBlur(gray, (3, 3), 0)
# cannyed = cv2.Canny(gaussed, 10, 220)
#
#
# # 将灰度边缘转化为BGR
# cannyed2 = cv2.cvtColor(cannyed, cv2.COLOR_GRAY2BGR)
#
#
# # 创建彩色边缘
# mask = cannyed > 0  # 边缘掩模
# canvas = np.zeros_like(img)
# canvas[mask] = img[mask]
#
#
# # 保存
# res = np.hstack((img, cannyed2, canvas))
# cv2.imwrite('result.jpg', res)
#
#
# # 显示
# cv2.imshow('canny in opencv', res)
#
#
# # 保持10s, 等待案件响应（超时或按键则进行下一步）
# key = 0xFF & cv2.waitKey(1000*10)
# if key in (ord('Q'), ord('q'), 27):
#     print('exiting!')
#
#
# # 销毁窗口
# cv2.destroyAllWindows()


from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import os, sys

#from utils import *


def mat2qpixmap(img):
    """ numpy.ndarray to qpixmap
    """
    height, width = img.shape[:2]
    if img.ndim == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim ==2:
        #qimage = QImage(img.flatten(), width, height, QImage.Format_Indexed8)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise Exception("Unstatistified image data format!")
    qimage = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap
    #qlabel.setPixmap(qpixmap)

class ImageView(QWidget):
    """显示 OpenCV 图片的 QWidget 控件
    """
    def __init__(self, winname = "ImageView"):
        super().__init__()
        self.setWindowTitle(winname)
        self.imageLabel = QLabel(self)
        self.imageLabel.setText(winname)
        #self.resize(200,150) # 宽W， 高H
        self.show()

    def setPixmap(self, img):
        #img = cv2.imread("test.png")
        if img is not None:
            H,W = img.shape[:2]
            qpixmap = mat2qpixmap(img)
            self.imageLabel.setPixmap(qpixmap)
            self.resize(W,H) # 宽W， 高H
            self.imageLabel.resize(W,H) # 宽W， 高H

class SliderCanny(QWidget):
    """创建滑动条控制界面，设置Canny边缘检测上下阈值。
    """
    def __init__(self, img=None):
        super().__init__()
        if img is None:
            raise Exception("The Image IS Empty !")
            pass
        self.img = img
        self.gray = cv2.GaussianBlur(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), (3,3), 0)
        self.setWindowTitle("Th")
        self.ImageView = ImageView("Source")
        self.CannyView = ImageView("Canny")
        self.ImageView.setPixmap(self.img)
        self.CannyView.setPixmap(self.img)

        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.addSliders()
        self.show()

    def addSliders(self):
        ## 创建水平滑动条
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider2 = QSlider(Qt.Horizontal)

        ## 设置范围和初值
        self.slider1.setRange(10, 250)
        self.slider2.setRange(10, 250)
        self.slider1.setValue(50)
        self.slider2.setValue(200)

        ## 添加到界面中
        self.mainLayout.addWidget(self.slider1)
        self.mainLayout.addWidget(self.slider2)

        ## 绑定信号槽
        self.slider1.valueChanged.connect(self.doCanny)
        self.slider2.valueChanged.connect(self.doCanny)

    def doCanny(self):
        th1 = self.slider1.value()
        th2 = self.slider2.value()
        print(th1, th2 )
        self.setWindowTitle("TH:{}~{}".format(th1, th2))

        ## Canny 边缘检测
        cannyed = cv2.Canny(self.gray, th1, th2)
        ## 创建彩色边缘
        mask = cannyed > 0                  # 边缘掩模
        canvas = np.zeros_like(self.img)    # 创建画布
        canvas[mask] = img[mask]            # 赋值边缘
        ## 显示结果
        self.CannyView.setPixmap(canvas)

if __name__ == "__main__":
    qApp = QApplication([])
    img = cv2.imread("004545.jpg")
    w2 = SliderCanny(img)
    sys.exit(qApp.exec_())
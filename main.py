import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import math

if __name__=='__main__':
    start=time.perf_counter()
    file_name='D:\FFOutput\\1600357086421.jpg'
    img_cv= cv2.imread(file_name)

    #把图像数组的正确读取（cv2本来是BGR）
    img_rgb=cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_cv)
    # plt.show()
    plt.imshow(img_rgb)
    plt.show()

    #输出参数，看看
    print('The type of the image:', type(img_cv))
    print('shape{}'.format(img_cv.shape))
    print('height: {}'.format(img_cv.shape[0]))
    print('the de of img_rgb:{}'.format(img_cv.shape[2]))
    print('max of img_rgb:{}'.format(img_cv.max()))
    #maxar=max(img_cv.all())
    #print('max of img_R:{}'.format(maxar))
    #要求每个分量里的最大值呢？

    #splite the RGB channel
    numtx1=np.zeros_like(img_cv)
    numtx2=np.zeros_like(img_cv)
    numtx3 = np.zeros_like(img_cv)
    numtx1[:,:,0]=img_rgb[:,:,0]
    numtx2[:,:,1]=img_rgb[:,:,1]
    numtx3[:,:,2] = img_rgb[:,:,2]

    img_gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
    #分离后的图显示
    # plt.imshow(img_gray)
    # plt.show()
    # plt.imshow(numtx1)
    # plt.show()
    # plt.imshow( numtx2)
    # plt.show()
    # plt.imshow( numtx3)
    # plt.show()

    #图片上加入文字
    # cv2.putText(img_cv, "yaodaoji", (200, 100), \
    #            cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    # plt.imshow(img_cv)
    # plt.show()

    #绘制灰度直方图。简单函数试试
    # x=np.linspace(-3,3,40)
    # y=2*x
    # y2=1/x
    #
    # plt.figure()
    # plt.plot(x,y)
    # plt.show()
    #
    # plt.figure()
    # plt.bar(x, y2)
    # plt.show()

    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        #老师教的办法。用起来
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
    cv2.waitKey()

    # #分别自己bar实现三通道直方图
    # plt.figure()
    # #把矩阵变成一维数组
    # mtx1r=numtx1.ravel()
    # #去掉零
    # b = np.argwhere(mtx1r==0)
    # mtx1r = np.delete(mtx1r, b)
    # hist =np.zeros(256)
    # #实现x，y（数组）
    # xaxis=np.linspace(0,255,256)
    # for i in range(255):
    #     mt1_tf=(mtx1r==i)
    #     hist[i]=np.sum(mt1_tf)
    # plt.title("red bar")
    # plt.bar(xaxis, hist)
    # plt.show()
    #
    # #第二个通道就照写，改一下数字
    # plt.figure()
    # mtx2r=numtx2.ravel()
    # b = np.argwhere(mtx2r==0)
    # mtx2r = np.delete(mtx2r, b)
    # hist =np.zeros(256)
    # xaxis=np.linspace(0,255,256)
    # for i in range(255):
    #     mt1_tf=(mtx2r==i)
    #     hist[i]=np.sum(mt1_tf)
    # plt.title("green bar")
    # plt.bar(xaxis, hist)
    # plt.show()
    #
    # plt.figure()
    # mtx3r=numtx3.ravel()
    # b = np.argwhere(mtx3r==0)
    # mtx3r = np.delete(mtx3r, b)
    # hist =np.zeros(256)
    # xaxis=np.linspace(0,255,256)
    # for i in range(255):
    #     mt1_tf=(mtx3r==i)
    #     hist[i]=np.sum(mt1_tf)
    # plt.title("blue bar")
    # plt.bar(xaxis, hist)
    # plt.show()





    #下面 本质还是 利用hist函数实现的。不算自己写的
    #比较运算速度那个我也还没做。
    plt.figure()
    mtx1r=numtx1.ravel()
    b = np.argwhere(mtx1r==0)
    mtx1r = np.delete(mtx1r, b)
    plt.hist(mtx1r, 256, [0, 256])
    plt.show()
    cv2.waitKey()

    plt.figure()
    mtx2r=numtx2.ravel()
    b = np.argwhere(mtx2r==0)
    mtx2r = np.delete(mtx2r, b)
    plt.hist(mtx2r, 256, [0, 256])
    plt.show()
    cv2.waitKey()

    plt.figure()
    mtx3r = numtx3.ravel()
    b = np.argwhere(mtx3r == 0)
    mtx3r = np.delete(mtx3r, b)
    plt.hist(mtx3r, 256, [0, 256])
    plt.show()
    cv2.waitKey()

    end=time.perf_counter()
    print("{}".format(end-start))









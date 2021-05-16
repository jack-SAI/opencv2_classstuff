import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__=='__main__':
    file_name='D:\FFOutput\\1600357086421.jpg'
    img_cv= cv2.imread(file_name)

    img_rgb=cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    plt.imshow(img_cv)
    plt.show()
    plt.imshow(img_rgb)
    plt.show()

    print('The type of the image:', type(img_cv))
    print('shape{}'.format(img_cv.shape))
    print('height: {}'.format(img_cv.shape[0]))
    print('the de of img_rgb:{}'.format(img_cv.shape[2]))
    print('max of img_rgb:{}'.format(img_cv.max()))
    #要求每个分量里的最大值呢？

    #nummat=ndarray(pres_img)
    #不知函数…
    #splite the RGB channel
    numtx1=np.zeros_like(img_cv)
    numtx2=np.zeros_like(img_cv)
    numtx3 = np.zeros_like(img_cv)
    numtx1[:,:,0]=img_rgb[:,:,0]
    numtx2[:,:,1]=img_rgb[:,:,1]
    numtx3[:,:,2] = img_rgb[:,:,2]

    img_gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray)
    plt.show()
    plt.imshow(numtx1)
    plt.show()
    plt.imshow( numtx2)
    plt.show()
    plt.imshow( numtx3)
    plt.show()

    cv2.putText(img_cv, "yaodaoji", (200, 100), \
               cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    plt.imshow(img_cv)
    plt.show()








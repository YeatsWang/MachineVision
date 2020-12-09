#1.导入所需要的库
import cv2    #导入OpenCV库
import matplotlib.pyplot as plt    #导入matplotlib库
import time    #导入时间库用于运行速度比较
%matplotlib inline

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #OpenCV载入默认是BGR，matplotlib需要使用RGB
    
#2.载入图片
test1 = cv2.imread('data/test1.jpg')
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)    #转成灰度图像给face detector
plt.imshow(gray_img, cmap='gray')    #用matplotlib的函数显示图片

                                     # 或者用OpenCV显示灰度图片
                                     # cv2.imshow('Test Imag', gray_img)
                                     # cv2.waitKey(0)
                                     # cv2.destroyAllWindows()
      
#3.载入HAAR联级分类器训练文件
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
#检测多尺度图片 (some images may be closer to camera than others)
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
print('Faces found: ', len(faces))    #打印找到的人脸数

#4.在彩色图像的人脸上画“绿色”矩形
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(convertToRGB(test1))    #转成RGB图片并显示

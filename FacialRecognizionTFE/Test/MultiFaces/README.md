# MultiFaces Comparator

## Face detector
This project permit the measure of perform Face Detector 

Face Detector used :
    
    - Haarcascade (from OpenCV)
        - haarcascade_frontalface_default.xml
    - Dnn (from OpenCV)
        - Model Coffee 
        - Model TensorFlow
    - HoG (from dlib)
        - dlib model
    - MMOD (from dlib)
        - Model : mmod_human_face_detector.dat
    - TinyFace (Custom Method)
        - Model : weight converted (hr_res101.mat)

### Face detector do not use :
    
    Haarcascade from haarcascade because a lot of false positives
    MMOD from dlib because it's very slow

### How to use FaceDetector class ?

    if __name__ == "__main__":
    
        # => Get the Face Detector Class
        fd = FaceDetector() 
        
        # => Get list of Images
        imgPath = list(paths.list_images("IMAGE_TO_DETECT")) 
        cpt = 0
        
        # => Check list content min 1 pictures
        if len(imgPath) > 1:
    
            for img in imgPath:
                img_read = cv2.imread(img)
                vframe = []
    
                cpt += 1
                print("[INFOS] Processing " + str(cpt) + "/" + str(len(imgPath)))
                
                # => Use your Prefered Face Detector
                vframe.append(cv2.resize(fd.detectFaceDlibHog(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
                
                vframe.append(cv2.resize(fd.detectFaceOpenCVDnn(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
                
                vframe.append(cv2.resize(fd.detectTinyFace(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
                
                vframe.append(cv2.resize(fd.detectFaceDlibMMOD(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
                
                # => show all result of face detectors
                Show_img(vframe)
                vframe.clear()

## TINY RESULT :

![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_1.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_19.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_21.jpg "Logo Title Text 1")


![alt text](https://github.com/SH4RKNANDO/MultiFaces/blob/master/IMG_RESULT/Result_22.jpg "Logo Title Text 1")


## TINY WAS NOT PERFECT !



## Requirements

Python library 
    
    dlib
    keras
    tensorflow-gpu or tensorflow
    opencv
    imutils
    numpy
    pickle
    scipy

### Nvidia driver

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.44       Driver Version: 440.44       CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 960M    Off  | 00000000:01:00.0 Off |                  N/A |
    | N/A   43C    P0    N/A /  N/A |      0MiB /  4046MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

### Nvidia cuda

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Oct_23_19:24:38_PDT_2019
    Cuda compilation tools, release 10.2, V10.2.89

### Nvidia cuda cudnn

download : https://developer.nvidia.com/cudnn
    
Archlinux :

    pacman -S cuda cudnn

## Optional Requirements

### intel mkl (math kernel library) 

source : https://software.intel.com/en-us/mkl
    
    yay -S intel-mkl
    yay -S intel-dnn
    yay -S tensorflow-cuda-mkl


### MXNET Neural Network
    
    yay -S mxnet-cuda-mkl-git

### Tutorial

PyImageSearch with ubuntu 18.04

https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/


## TODO:

    Optimizing TinyFace


## Developper Info

    Author : Jordan Bertieaux
    Version: 1.0
    OS : archlinux
    Kernel : 5.4.11-arch1-1 #1 SMP PREEMPT Sun, 12 Jan 2020 12:15:27 +0000 x86_64 GNU/Linux
    Python version 3
   
## Thanks

    PyImageSearch : https://www.pyimagesearch.com/
    Cyndonia : https://github.com/cydonia999/Tiny_Faces_in_Tensorflow

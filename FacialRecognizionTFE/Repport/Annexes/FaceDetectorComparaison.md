# Détecteurs de visages (comparaison)

## 1.0 Haar Cascade dans OpenCV

### Introduction

Le détecteur de visage basé sur Haar Cascade était le dernier cri en matière de détection de visage 
depuis de nombreuses années depuis 2001, année de son introduction par Viola et Jones. 
Il y a eu de nombreuses améliorations ces dernières années. 
OpenCV a beaucoup de modèles basés sur Haar qui peuvent être trouvés ici

https://github.com/opencv/opencv/tree/master/data/haarcascades

### Fonctionnement

L'extrait de code ci-dessus charge le fichier de modèle haar cascade et l'applique à une image en niveaux de gris.
la sortie est une liste contenant les visages détectés. Chaque membre de la liste
est à nouveau une liste de 4 éléments indiquant les coordonnées (x, y) du coin supérieur gauche ainsi que
la largeur et la hauteur de la face détectée.

### Avantages/Inconvénient

- Avantages
    - Fonctionne presque en temps réel sur le processeur
	- Architecture simple
	- Détecte les visages à différentes échelles
- Inconvénients
	- L'inconvénient majeur de cette méthode est qu'elle donne beaucoup de fausses prédictions.
	- Ne fonctionne pas sur les images non frontales.
	- Ne fonctionne pas sous occlusion


#############################################################################


## 2.0 DNN dans OpenCV

### introduction

Ce modèle a été inclus dans OpenCV à partir de la version 3.3. 
Il est basé sur un détecteur Single-Shot-Multibox,
utilise l’architecture ResNet-10 comme infrastructure.
 
Le modèle a été formé à l'aide d'images disponibles sur le Web, mais la source n'est pas révélée. 
OpenCV fournit 2 modèles pour ce détecteur de visage.

- caffe (float 16 bits)
- Tensorflow (quantifiée 8 bits)

### Utilisation

Exemple d'implémentations :

    DNN = "TF"
    if DNN == "CAFFE":
        modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "opencv_face_detector_uint8.pb"
        configFile = "opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)



Nous chargeons le modèle requis en utilisant le code ci-dessus.
Si nous voulons utiliser le modèle à virgule flottante de Caffe, nous utilisons les fichiers caffemodel et prototxt.
Sinon, nous utilisons le modèle de tensorflow quantifié. Notez également la différence dans la façon dont nous lisons
les réseaux pour Caffe et Tensorflow.


    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
     
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

Dans le code ci-dessus, l'image est convertie en blob et transmise à travers le réseau à l'aide de la fonction forward ().
Les détections de sortie sont une matrice 4-D, où La 3ème dimension itère sur les visages détectés. (i est l'itérateur sur le nombre de faces).
La quatrième dimension contient des informations sur le cadre de sélection et le score de chaque visage. Par exemple, les détections [0,0,0,2] donnent le score de confiance du premier visage et les détections [0,0,0,3: 6] le cadre de sélection.


Les coordonnées de sortie du cadre de sélection sont normalisées entre [0,1]. 
Ainsi, les coordonnées doivent être multipliées par la hauteur et la largeur de l'image d'origine pour obtenir 
le cadre de sélection correct sur l'image.

### Avantages 

- Avantages
    - La plus précise des quatre méthodes
    - Fonctionne en temps réel sur le processeur
    - Fonctionne pour différentes orientations du visage - haut, bas, gauche, droite, côté, etc.
    - Fonctionne même sous occlusion importante
    - Détecte les visages à différentes échelles (détecte les visages grands et petits)
- Inconvénients
	- Pas inconvénient majeur pour cette méthode
	

Le détecteur basé sur DNN surmonte tous les inconvénients du détecteur à cascade de Haar, 
sans compromettre aucun avantage fourni par Haar. 
Nous ne voyions aucun inconvénient majeur pour cette méthode,
si ce n’est qu’elle est plus lente que le Détecteur de visage basé sur Dlib HoG dont il est question ci-après.

Il serait prudent de dire qu'il est temps de faire ses adieux au détecteur de visage basé sur Haar et que le détecteur 
de visage basé sur DNN devrait être le choix préféré d'OpenCV.



#############################################################################


## HoG dans Dlib

### introduction

Il s'agit d'un modèle de détection de visage largement utilisé, basé sur les fonctionnalités HoG et le SVM.
Vous pouvez en savoir plus sur HoG dansnotre post. Le modèle est construit à partir de 5 filtres HOG: à l'avant,
à gauche, à droite, à l'avant mais pivoté à gauche et à l'avant, mais pivoté à droite.
Le modèle est intégré au fichier d’ en- tête même.

Le jeu de données utilisé pour la formation comprend 2825 images obtenues à partir du jeu de données LFW
et annotées manuellement par Davis King, l'auteur de Dlib. Il peut être téléchargé à partir d' ici .


    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(frameDlibHogSmall, 0)
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()



Dans le code ci-dessus, nous chargeons d'abord le détecteur de visage. 
Ensuite, nous passons l'image à travers le détecteur. 
Le deuxième argument est le nombre de fois que nous voulons agrandir l'image.
Plus vous êtes haut de gamme, meilleures sont vos chances de détecter des visages plus petits.
Cependant, la mise à l'échelle de l'image aura un impact considérable sur la vitesse de calcul.
La sortie se présente sous la forme d'une liste de faces avec les coordonnées (x, y) des coins diagonaux.


- Avantages
    - Méthode la plus rapide sur le processeur
    - Fonctionne très bien pour les faces frontales et légèrement non frontales
    - Modèle léger par rapport aux trois autres.
    - Fonctionne sous petite occlusion


Fondamentalement, cette méthode fonctionne dans la plupart des cas sauf quelques-uns, comme indiqué ci-dessous.


Les inconvénients

- L'inconvénient majeur est qu'il ne détecte pas les petits visages, car il est conçu pour une taille minimale de 80 × 80. 
Ainsi, vous devez vous assurer que la taille du visage doit être supérieure à celle de votre application. 
Vous pouvez toutefois former votre propre détecteur de visage pour les visages de petite taille.

- La boîte englobante exclut souvent une partie du front et même une partie du menton parfois.
- Ne fonctionne pas très bien sous occlusion importante
- Ne fonctionne pas pour les faces latérales et les faces extrêmes non frontales, comme regarder vers le bas ou le haut.


#############################################################################


4. Détecteur de visage CNN dans Dlib
Cette méthode utilise un détecteur d’objets à marge maximale (MMOD) avec des fonctionnalités basées sur CNN. 
Le processus de formation pour cette méthode est très simple et vous n'avez pas besoin d'une grande quantité
 de données pour former un détecteur d'objet personnalisé. Pour plus d'informations sur la formation, 
 visitez le site Web .

Le modèle peut être téléchargé à partir du référentiel dlib-models .
Il utilise un ensemble de données étiqueté manuellement par son auteur, Davis King, 
et constitué d'images provenant de différents ensembles de données, tels que ImageNet, PASCAL VOC, VGG, WIDER et 
Face Scrub. Il contient 7220 images. Le jeu de données peut être téléchargé à partir d' ici



    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
    faceRects = dnnFaceDetector(frameDlibHogSmall, 0)
    for faceRect in faceRects:
        x1 = faceRect.rect.left()
        y1 = faceRect.rect.top()
        x2 = faceRect.rect.right()
        y2 = faceRect.rect.bottom()



Le code est similaire au détecteur HoG, sauf que dans ce cas, nous chargeons le modèle de détection de visage CNN. 
De plus, les coordonnées sont présentes dans un objet rect.

- Avantages
    -Fonctionne pour différentes orientations du visage
    - Robuste à l'occlusion
    - Fonctionne très vite sur le GPU
    - Processus de formation très facile 
- Les inconvénients
    - Très lent sur le processeur

Ne détecte pas les petits visages, car il est entraîné pour une taille minimale de 80 × 80. Ainsi, 
vous devez vous assurer que la taille du visage doit être supérieure à celle de votre application.
Vous pouvez toutefois former votre propre détecteur de visage pour les visages de petite taille.
La boîte englobante est encore plus petite que le détecteur HoG.

5. Comparaison de précision

J'ai essayé d'évaluer les 4 modèles à l'aide du jeu de données FDDB à l'aide du script utilisé pour évaluer 
le modèle OpenCV-DNN . Cependant, j'ai trouvé des résultats surprenants. Dlib avait des nombres pires que Haar,
bien que les sorties de dlib soient bien meilleures. Ci-dessous sont les scores de précision pour les 4 méthodes.


#############################################################################

## Comparaison des Modèle (pratique)

### Classe FaceDetector

    from __future__ import division
    import cv2
    import dlib
        
    class FaceDetector:
        def __init__(self):
            # OpenCV HAAR
            self._faceCascade = cv2.CascadeClassifier('Data/Model/haarcascade_frontalface_default.xml')

        # OpenCV DNN supports 2 networks.
        # 1. FP16 version of the original caffe implementation ( 5.4 MB )
        # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
        DNN = "TF"

        if DNN == "CAFFE":
            self._modelFile = "Data/Model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            self._configFile = "Data/Model/deploy.prototxt"
            self._net = cv2.dnn.readNetFromCaffe(self._configFile, self._modelFile)
        else:
            self._modelFile = "Data/Model/opencv_face_detector_uint8.pb"
            self._configFile = "Data/Model/opencv_face_detector.pbtxt"
            self._net = cv2.dnn.readNetFromTensorflow(self._modelFile, self._configFile)

        self._conf_threshold = 0.8

        # DLIB HoG
        self._hogFaceDetector = dlib.get_frontal_face_detector()

        # DLIB MMOD
        self._dnnFaceDetector = dlib.cnn_face_detection_model_v1("Data/Model/mmod_human_face_detector.dat")

    def detectFaceOpenCVHaar(self, frame, inHeight=300, inWidth=0):
        frameOpenCVHaar = frame.copy()
        frameHeight = frameOpenCVHaar.shape[0]
        frameWidth = frameOpenCVHaar.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
        frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

        faces = self._faceCascade.detectMultiScale(frameGray)
        bboxes = []
        cv2.putText(frameOpenCVHaar, "OpenCV HaarCascade", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                    cv2.LINE_AA)

        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                      int(x2 * scaleWidth), int(y2 * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        return frameOpenCVHaar

    def detectFaceOpenCVDnn(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

        self._net.setInput(blob)
        detections = self._net.forward()
        bboxes = []
        cv2.putText(frameOpencvDnn, "OpenCV DNN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self._conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        return frameOpencvDnn

    def detectFaceDlibHog(self, frame, inHeight=300, inWidth=0):

        frameDlibHog = frame.copy()
        frameHeight = frameDlibHog.shape[0]
        frameWidth = frameDlibHog.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

        frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
        faceRects = self._hogFaceDetector(frameDlibHogSmall, 0)
        print(frameWidth, frameHeight, inWidth, inHeight)
        bboxes = []
        cv2.putText(frameDlibHog, "OpenCV HoG", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        for faceRect in faceRects:
            cvRect = [int(faceRect.left() * scaleWidth), int(faceRect.top() * scaleHeight),
                      int(faceRect.right() * scaleWidth), int(faceRect.bottom() * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        return frameDlibHog

    def detectFaceDlibMMOD(self, frame, inHeight=300, inWidth=0):

        frameDlibMMOD = frame.copy()
        frameHeight = frameDlibMMOD.shape[0]
        frameWidth = frameDlibMMOD.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))

        frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)
        faceRects = self._dnnFaceDetector(frameDlibMMODSmall, 0)

        print(frameWidth, frameHeight, inWidth, inHeight)
        bboxes = []

        cv2.putText(frameDlibMMOD, "OpenCV MMOD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        for faceRect in faceRects:
            cvRect = [int(faceRect.rect.left() * scaleWidth), int(faceRect.rect.top() * scaleHeight),
                      int(faceRect.rect.right() * scaleWidth), int(faceRect.rect.bottom() * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameDlibMMOD, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        return frameDlibMMOD

### Implémentation de la classe














1) Installation des déendances
sudo pacman -S cmake python-numpy python-pip opencv
sudo pip install imutils

2) Installation de la librairies dlib

sudo pip install dlib

Chap 1 : Choix du moteur de reconnaissance




Analyse les caractéristiques du visage (DLIB)

DLIB : Avantage
  
  -> Exploitez facilement toutes les fonctionnalités de Python + dlib (détection de visage, repères de visage, suivi de corrélation, etc.)…
  -> Avec moins de dépendances et un processus d'installation plus facile.
  -> Analyse les traits de Caractéritiques du visage 



Fonctionnement : XXX
AV/INC : XXX


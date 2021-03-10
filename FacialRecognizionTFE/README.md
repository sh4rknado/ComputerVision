# FacialRecognizionTFE
TFE of FacialRecognizion

## Configuration

### file : FacialRecognizion/Data/Config/detector.ini

	# *=================================================================================*
	# |                      S E C T I O N    G E N E R A L                             |
	# *=================================================================================*
	[General]

	# *--------------------------------*
	# | use ALPR PlateRecognizer       |
	# | use facial Detecor process     |
	# | use facial recognizion process |
	# *--------------------------------*
	use_alpr=True
	use_facial_recognizion=True

	# *------------------------------*
	# | Default FaceDetector Process |
	# | Set use_face_detector=True   |
	# *------------------------------*
	# use DNN | Haar | MMOD | HoG | Tiny
	face_detector_process=Tiny

	# *=================================================================================*
	# |                      S E C T I O N    F a c e D e t e c t o r                   |
	# *=================================================================================*
	#
	# *==================================*
	# |     SECTION FaceDetectorDNN      |
	# *==================================*
	[FaceDetectorDNN]
	#
	# *---------------------------------*
	# | Select the model [TF OR CAFFE]  |
	# *---------------------------------*
	process_model=CAFFE

	# *------------------*
	# |   Model CAFFEE   |
	# *------------------*
	# modelFile=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/FaceDetector/res10_300x300_ssd_iter_140000_fp16.caffemodel
	# configFile=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/FaceDetector/deploy.prototxt

	# *------------------*
	# | Model TensorFlow |
	# *------------------*
	modelFile=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/FaceDetector/opencv_face_detector_uint8.pb
	configFile=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/FaceDetector/opencv_face_detector.pbtxt

	# *-------------------------------------*
	# | precision of detector (Default:0.7) |
	# | Most Poeple doesn't touch this      |
	# *-------------------------------------*
	conf_threshold=0.7


	# *==================================*
	# |     SECTION FaceDetectorHaar     |
	# *==================================*
	[FaceDetectorHaar]
	# *-------------------------------*
	# | Select the haarcascades model |
	# *-------------------------------*
	haarcascade_frontalface_default=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/FaceDetector/haarcascade_frontalface_default.xml

	# *-----------------------------------*
	# | Delete the fake face Detection    |
	# | Minimum Resize MultiScale (float) |
	# | Maximum Resize MultiScale (int)   |
	# *-----------------------------------*
	min_multiscale=1.3
	max_multiscale=3

	# *==================================*
	# |     SECTION FaceDetectorHoG      |
	# *==================================*
	[FaceDetectorHoG]
	# Nothing to choose

	# *==================================*
	# |    SECTION FaceDetectorMMOD      |
	# *==================================*
	[FaceDetectorMMOD]
	# select the model
	cnn_face_detection_model_v1=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/FaceDetector/mmod_human_face_detector.dat

	# *==================================*
	# |    SECTION FaceDetectorTiny      |
	# *==================================*
	[FaceDetectorTiny]
	# * -------------------*
	# |   Default Config   |
	# * -------------------*
	MAX_INPUT_DIM=5000.0
	prob_thresh=0.5
	nms_tresh=0.1
	lw = int(3)

	# *----------------------* 
	# | model RESNET10       |
	# | converted to weight  |
	# *----------------------*
	model=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/hr_res101.weight

	# *=================================================================================*
	# |        S E C T I O N    O B J E  C  T    D E T E C T I O N   Y O L O V 3        |
	# *=================================================================================*
	[object]

	# Pattern for match detection
	detect_pattern=(person|car|motorbike|truck|bicycle|handbag|suitcase|baseball bat|bottle|knife)

	# minimum probability to filter weak detections
	confidence=0.5

	# threshold when applying non-maximal suppression
	threshold=0.3

	# Yolo class labels
	yolo_labels_path=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/yolo/coco.names

	# Yolo config
	yolo_config_path=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/yolo/yolov3.cfg

	# Yolo weights
	yolo_weights_path=/home/zerocool/PycharmProjects/FacialRecognizionTFE/FacialRecognizion/Data/Model/yolo/yolov3.weights

	# Yolo Add Rect with name into ZM
	yolo_override_ZM=True

	# Yolo Show Percent Confidences
	yolo_show_percent=False

	# Delete the static element
	save_temporary=True


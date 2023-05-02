from deepface import DeepFace

# DeepFace.stream("dataset") #opencv
# DeepFace.stream("dataset", detector_backend = 'opencv')
#DeepFace.stream("dataset", detector_backend = 'ssd')
DeepFace.stream("dataset", detector_backend = 'mtcnn',enable_face_analysis=False,time_threshold=1,frame_threshold=1)
#DeepFace.stream("dataset", detector_backend = 'dlib')
#DeepFace.stream("dataset", detector_backend = 'retinaface')

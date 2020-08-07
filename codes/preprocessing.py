import os
import cv2
class Image():
    def __init__(self, name, file_name, dir_path):
        self.name = name
        self.file_name = file_name
        self.dir_path = dir_path




'''input_dir : the directory of the LFW dataset
   output_dir : the directory that the processed images are stored in 
   opencv-4.2.0 directory is needed
   LFW dataset is needed
   run the code, the face will be detected and intercepted out, then stored in the output_dir
'''

# create the new directory which preprocessed images are stored in
input_dir = 'D:\\UF course files\\pattern recognition\\project\\facenet\\lfw'
output_dir = 'D:\\UF course files\\pattern recognition\\project\\facenet\\lfw1'
if os.path.exists(output_dir)!=1:
        os.makedirs(output_dir)
# create name classes
classes = []
for name in os.listdir(input_dir):
        class_dir = input_dir + '\\' + name
        faces = os.listdir(class_dir)
        classes.append(Image(name,faces,class_dir))

'''detect the face of each image and store the detected face into output_dir'''
findface = cv2.CascadeClassifier('opencv-4.2.0\\data\\haarcascades_cuda\\haarcascade_frontalface_alt.xml')
for name in classes:
    # create a directory for one name
    name_dir_write = output_dir + '\\' + name.name
    if os.path.exists(name_dir_write) != 1:
        os.makedirs(name_dir_write)
    # one directory may has many images
    for face_file in name.file_name:
        #read one image
        file_path_read = name.dir_path+'\\'+face_file
        img = cv2.imread(file_path_read)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #detect the face
        face_detected = findface.detectMultiScale(gray_img,1.3,5)
        for f_x, f_y, f_w, f_h in face_detected:
            face_img = gray_img[f_y:f_y+f_h, f_x:f_x+f_w]
            # resize the face img
            resized_face = cv2.resize(face_img, (200, 200))
            resized_face_path = name_dir_write+'\\'+face_file+ '.jpg'
            cv2.imwrite(resized_face_path, resized_face)

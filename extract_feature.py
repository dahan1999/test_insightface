# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
from models import base_server
from configs import configs
from pathlib import Path
import pickle

# source directory for the face photos, in .jpg
src_dir_in_str = "/Users/hmeng/github/tf-insightface/tests/faces.20220113/avatar"

# Define a Base Server
srv = base_server.BaseServer(model_fp=configs.face_describer_model_fp,
                             input_tensor_names=configs.face_describer_input_tensor_names,
                             output_tensor_names=configs.face_describer_output_tensor_names,
                             device=configs.face_describer_device)

# face database, a face file name and its feature array
face_db = {}

def extract_feature(face_file):
    try:
        # Read example image
        face_img = cv2.imread(face_file)
        face_img = cv2.resize(face_img, configs.face_describer_tensor_shape)

        # Define input tensors feed to session graph
        dropout_rate = 0.5
        input_data = [np.expand_dims(face_img, axis=0), dropout_rate]

        # Run prediction
        #prediction = srv.inference(data=input_data)
        return srv.inference(data=input_data)

        # Print results
        # print('512-D Features are \n{}'.format(prediction))

    except OSError as err:
        print("OS error: {0}".format(err))


pathlist = Path(src_dir_in_str).rglob('*.jpg')
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    feature = extract_feature(path_in_str)
    face_db[path_in_str] = feature

# create a binary pickle file
f = open("file.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(face_db,f)

# close file
f.close()

print("Done")
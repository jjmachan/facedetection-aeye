import os
from random import sample
import pickle

from .models.mtcnn import MTCNN
from .models.inception_resnet_v1 import InceptionResnetV1

class FaceDetection():
    def __init__(self, saved_models=None, mtcnn=None, inception_net=None):

        # path of pretrained weights
        self.saved_models = saved_models
        self.facedb = FaceDatabase

        print(self.saved_models)

        # in models are not passed through init, load it
        # from the saved_models dir.
        if mtcnn is not None:
            self.mtcnn = mtcnn
        else:
            self.mtcnn = MTCNN()

        if inception_net is not None:
            self.inception_net = inception_net
        else:
            self.inception_net = InceptionResnetV1(pretrained='vggface2',
                                                   path=self.saved_models)

    def detect_faces(self, img):
        """
        Detect all faces in PIL image and return the
        bounding boxes and facial landmarks
        """

        boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
        return boxes, probs, points


    def add_person(self, name, img):
        """
        Add a person for detection
        """


class FaceDatabase():
    """
    Class that acts as the database to store
    the saved aligned faces
    """
    def __init__(self, db_file=None):
        if db_file is None:
            print('init new db')
            self.db_file = './faces_db.fdb'
            self.database = dict()
        else:
            print('loading db...')
            self.db_file = db_file
            self._load()

    def _save(self):
        with open(self.db_file, 'wb') as f:
            pickle.dump(self.__dict__, f)
            print('saved db')

    def _load(self):
        with open(self.db_file, 'rb') as f:
            state_dict = pickle.load(f)
            self.__dict__.update(state_dict)
            print(f'loaded {self.db_file}')

    def add(self, name, img_tensor):
        if name in self.database:
            self.database[name].append(img_tensor)
            print(f'Face {name} updated with new image')
        else:
            self.database[name] = [img_tensor]
            print(f'New face {name} added!')

    def get_batched(self):
        all_faces = list()
        for people in self.database.keys():
            faces = self.database[people]
            if len(faces) < 5:
                continue
            all_faces.extend(sample(faces, 5))
        return all_faces

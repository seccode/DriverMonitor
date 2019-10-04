import cv2
import dlib
import numpy as np
from imutils import face_utils
import argparse
import pygame
import time

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load('beep-07.mp3')

parser = argparse.ArgumentParser(description="Pass video file")
parser.add_argument("--video",dest="video",default="0",
                    help="Path to video file")
args = parser.parse_args()

class FaceDetector:
    def __init__(self):
        self.dist_coeffs = np.zeros((4,1))
        self.face_landmark_path = './shape_predictor_68_face_landmarks.dat'
        self.eyes_area = []

    def _get_full_model_points(self,filename='assets/model.txt'):
            raw_value = []
            with open(filename) as file:
                for line in file:
                    raw_value.append(line)
            model_points = np.array(raw_value, dtype=np.float32)
            model_points = np.reshape(model_points, (3, -1)).T

            model_points[:, 2] *= -1
            return model_points

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
            point_3d = []
            rear_size = 75
            rear_depth = 0
            point_3d.append((-rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, -rear_size, rear_depth))

            front_size = 120
            front_depth = 100
            point_3d.append((-front_size, -front_size, front_depth))
            point_3d.append((-front_size, front_size, front_depth))
            point_3d.append((front_size, front_size, front_depth))
            point_3d.append((front_size, -front_size, front_depth))
            point_3d.append((-front_size, -front_size, front_depth))
            point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

            (point_2d, _) = cv2.projectPoints(point_3d,
                                              rotation_vector,
                                              translation_vector,
                                              self.camera_matrix,
                                              self.dist_coeffs)
            point_2d = np.int32(point_2d.reshape(-1, 2))

            cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
            cv2.line(image, tuple(point_2d[1]), tuple(
                point_2d[6]), color, line_width, cv2.LINE_AA)
            cv2.line(image, tuple(point_2d[2]), tuple(
                point_2d[7]), color, line_width, cv2.LINE_AA)
            cv2.line(image, tuple(point_2d[3]), tuple(
                point_2d[8]), color, line_width, cv2.LINE_AA)

    def get_head_pose(self,shape):
        image_pts = np.float32([shape])
        _, rotation_vec, translation_vec = cv2.solvePnP(self.model_points_68,
                                                        image_pts,
                                                        self.camera_matrix,
                                                        self.dist_coeffs)
        return rotation_vec, translation_vec

    def eyes_open(self,shape):
        '''
        Heuristic to determine if eyes are open based on a continuously
        calculated average eye area
        '''
        eye_left = np.array([shape[36],shape[37],shape[38],
                            shape[39],shape[40],shape[41]])
        eye_right = np.array([shape[42],shape[43],shape[44],
                            shape[45],shape[46],shape[47]])

        left_area = self.shoelace_formula(eye_left)
        right_area = self.shoelace_formula(eye_right)
        self.eyes_area.append(left_area)
        self.eyes_area.append(right_area)
        if (left_area+right_area)/2 < (np.mean(self.eyes_area)/1.5):
            return False
        return True

    def shoelace_formula(self,points):
        '''Determine area of polygon from points'''
        x = points[:,0]
        y = points[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


    def detect(self):
        if args.video == '0':
            args.video = 0
        cap = cv2.VideoCapture(args.video)
        assert cap.isOpened(), "Video {} not found".format(args.video)

        _, test_frame = cap.read()

        size = test_frame.shape

        self.model_points_68 = self._get_full_model_points()

        self.focal_length = size[1]
        self.camera_center = (size[1] / 2, size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.face_landmark_path)

        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
            if ret:
                face_rects = self.detector(frame, 0)
                if len(face_rects) > 0:
                    shape = face_utils.shape_to_np(self.predictor(frame, face_rects[0]))
                    print(self.eyes_open(shape))
                    for i, item in enumerate(shape):
                        if i >= 36 and i <= 47:
                            cv2.circle(frame,tuple(item),1,(0,255,0),-1)
                        else:
                            cv2.circle(frame,tuple(item),1,(255,0,0),-1)

                    rotation_vec, translation_vec = self.get_head_pose(shape)
                    self.draw_annotation_box(frame,rotation_vec,translation_vec)
                else:
                    pygame.mixer.music.play(1,0.0)
                    time.sleep(.02)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) == 27:
                    break

if __name__ == '__main__':
    FaceDetector().detect()

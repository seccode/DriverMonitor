import cv2
import dlib
import numpy as np
from imutils import face_utils
import argparse
import pygame
import time

class FaceDetector:
    def __init__(self):
        self.dist_coeffs = np.zeros((4,1))
        self.face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
        self.eyes_area = [] # Store area of eye feature
        self.head_rotation = [] # Store head rotation information

    def get_full_model_points(self,filename='model.txt'):
            raw_value = []
            with open(filename) as file:
                for line in file:
                    raw_value.append(line)
            model_points = np.array(raw_value, dtype=np.float32)
            model_points = np.reshape(model_points, (3, -1)).T
            model_points[:, 2] *= -1
            return model_points

    def draw_annotation_box(self, image,
                                shape,
                                rotation_vector,
                                translation_vector,
                                color=(255, 255, 255),
                                line_width=2):
            # Draw 3D box on frame in front of head
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

            # Project 3D points onto 2D frame
            (point_2d, _) = cv2.projectPoints(point_3d,
                                              rotation_vector,
                                              translation_vector,
                                              self.camera_matrix,
                                              self.dist_coeffs)

            point_2d = np.int32(point_2d.reshape(-1, 2))

            head_turned = False
            if point_2d[0][0] < point_2d[5][0] or \
                point_2d[0][1] < point_2d[5][1] or \
                point_2d[2][0] > point_2d[7][0] or \
                point_2d[2][1] > point_2d[7][1]:
                head_turned = True
                color = (0,255,0)

            # Add lines to frame
            cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
            cv2.line(image, tuple(point_2d[1]), tuple(
                point_2d[6]), color, line_width, cv2.LINE_AA)
            cv2.line(image, tuple(point_2d[2]), tuple(
                point_2d[7]), color, line_width, cv2.LINE_AA)
            cv2.line(image, tuple(point_2d[3]), tuple(
                point_2d[8]), color, line_width, cv2.LINE_AA)

            # Add feature points to head
            for i, item in enumerate(shape):
                if i >= 36 and i <= 47: # Eye feature points
                    cv2.circle(image,tuple(item),1,(0,255,0),-1)
                else:
                    cv2.circle(image,tuple(item),1,(255,0,0),-1)

            return head_turned

    def get_head_pose(self,shape):
        # Find rotational and translational vectors of head position
        image_pts = np.float32([shape])
        _, rotation_vec, translation_vec = cv2.solvePnP(self.model_points_68,
                                                        image_pts,
                                                        self.camera_matrix,
                                                        self.dist_coeffs)

        return (rotation_vec, translation_vec)

    def eyes_area_per_face_area(self,shape):
        # Heuristic to determine if eyes are open based on a continuously
        # calculated average eye area controlled for head size (proxy for depth)
        eye_left = np.array([shape[36],shape[37],shape[38],
                            shape[39],shape[40],shape[41]])
        eye_right = np.array([shape[42],shape[43],shape[44],
                            shape[45],shape[46],shape[47]])

        face_outline = np.array([shape[1],shape[9],shape[17]])
        face_area = self.shoelace_formula(face_outline)

        left_area = self.shoelace_formula(eye_left) / face_area
        right_area = self.shoelace_formula(eye_right) / face_area
        self.eyes_area.append(left_area)
        self.eyes_area.append(right_area)
        return (left_area+right_area)/2

    def shoelace_formula(self,points):
        # Determine area of polygon from points
        x = points[:,0]
        y = points[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    def detect(self):
        # Read from video and detect head position
        if args.video == '0':
            args.video = 0
        cap = cv2.VideoCapture(args.video)
        assert cap.isOpened(), "Video {} not found".format(args.video)

        # Find size of frame
        _, test_frame = cap.read()
        size = test_frame.shape
        self.focal_length = size[1]
        self.camera_center = (size[1] / 2, size[0] / 2)

        # 68 facial feature points
        self.model_points_68 = self.get_full_model_points()

        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.face_landmark_path)

        # Matrix with shape 30 x 6 that stores past 30 frames where 3 features
        # are 1. Face detected, 2. Eyes Area / Face Area, 3. Head Rotation Bool
        # 4-6. Head Rotation,
        driver_state = np.full(shape=(30,6),fill_value=None)

        y_padding_bottom = 10
        y_padding_top = 10
        x_padding_left = 10
        x_padding_right = 10

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame for faster detection
            frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
            # frame = frame[y_padding_top:frame.shape[1]-y_padding_top-y_padding_bottom,
                        # x_padding_left:frame.shape[0]-x_padding_left-x_padding_right]

            current_state = np.array([[0,0,0,0,0,0]])

            head_turned = False
            # Detect face in frame
            face_rects = self.detector(frame, 0)
            if len(face_rects) > 0: # Head detected
                # driver_state['Face'] = True
                # Get head feature points from first detected head
                shape = face_utils.shape_to_np(self.predictor(frame, face_rects[0]))

                # Find position of head
                rotation_vec, translation_vec = self.get_head_pose(shape)

                # Check if eyes are open
                eyes_area = self.eyes_area_per_face_area(shape)

                # Draw facial features
                head_turned = self.draw_annotation_box(frame,shape,rotation_vec,translation_vec)

                # Update current driver state
                current_state = np.append(np.array([1,eyes_area,(1 if head_turned else 0)]),translation_vec.flatten()).reshape(1,6)

            # Update driver state feature vector
            driver_state = np.concatenate((np.delete(driver_state,0,axis=0),
                                            current_state),axis=0)

            # Use heuristic of feature vector to determine if inattentive
            if driver_state[0][0] != None and \
                (
                # (np.max(driver_state[-10:,0]) == 0) or \
                # (np.mean(driver_state[:,1]) < (np.mean(self.eyes_area) / 2)) or \
                (np.mean(driver_state[-10:,2]) > 0.9) or \
                (False)
                ):
                # Play alert sound
                pygame.mixer.music.play(1,0.0)
                time.sleep(.02)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) == 27:
                break

if __name__ == '__main__':
    # Load alert sound
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load('beep-07.mp3')

    # Take video argument, default to webcam
    parser = argparse.ArgumentParser(description="Pass video file")
    parser.add_argument("--video",dest="video",default="0",
                        help="Path to video file")
    parser.add_argument("--show",dest="show",default=1,
                        help="1 to show frame, 0 to not show frame")
    args = parser.parse_args()

    FaceDetector().detect()

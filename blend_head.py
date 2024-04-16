import cv2
import logging
import mediapipe as mp
import math
import numpy as np
import os
import sys
import time
import torch

def _get_content_pose_landmarks(input_frames=None):
    boxes = []  # bounding boxes of human face
    all_pose_landmarks, all_content_landmarks = [], []  # content landmarks include lip and jaw landmarks

    _fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                            min_detection_confidence=0.5)
    
    for frame_idx, full_frame in enumerate(input_frames):
        h, w = full_frame.shape[0], full_frame.shape[1]
        
        results = _fm.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            _msg = 'E001: Face Detection failed at frame %05d. Falling back to initial estimate' % frame_idx
            logging.info(_msg)

        else:
            face_landmarks = results.multi_face_landmarks[0]
    
            # (1)get the marginal landmarks to crop face
            x_min, x_max, y_min, y_max = 999, -999, 999, -999
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in self.all_landmarks_idx:
                    if landmark.x < x_min:
                        x_min = landmark.x
                    if landmark.x > x_max:
                        x_max = landmark.x
                    if landmark.y < y_min:
                        y_min = landmark.y
                    if landmark.y > y_max:
                        y_max = landmark.y
            ##########plus some pixel to the marginal region##########
            #note:the landmarks coordinates returned by mediapipe range 0~1
            plus_pixel = 25
            x_min = max(x_min - plus_pixel / w, 0)
            x_max = min(x_max + plus_pixel / w, 1)

            y_min = max(y_min - plus_pixel / h, 0)
            y_max = min(y_max + plus_pixel / h, 1)
            y1, y2, x1, x2 = int(y_min * h), int(y_max * h), int(x_min * w), int(x_max * w)
            boxes.append([y1, y2, x1, x2])
            
    boxes = np.array(boxes)

    face_crop_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] \
                            for image, (y1, y2, x1, x2) in zip(input_frames, boxes)]

    # (3)detect facial landmarks
    for frame_idx, full_frame in enumerate(input_frames):
        h, w = full_frame.shape[0], full_frame.shape[1]
        results = _fm.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            _msg = 'E002: Face Detection failed at frame %05d. Falling back to initial estimate' % frame_idx
            logging.info(_msg)
            # Using previous detections
            _prev = all_pose_landmarks[-1]
            all_pose_landmarks.append(_prev)
            _prev = all_content_landmarks[-1]
            all_content_landmarks.append(_prev)
        else:
            face_landmarks = results.multi_face_landmarks[0]

            _pl, _cl = [], []
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in self.pose_landmark_idx:
                    _pl.append((idx, w * landmark.x, h * landmark.y))
                if idx in self.content_landmark_idx:
                    _cl.append((idx, w * landmark.x, h * landmark.y))

            # normalize landmarks to 0~1
            y_min, y_max, x_min, x_max = face_crop_results[frame_idx][1]  #bounding boxes
            _pl = [[idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in _pl]
            _cl = [[idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in _cl]
            all_pose_landmarks.append(_pl)
            all_content_landmarks.append(_cl)

    return all_pose_landmarks, all_content_landmarks, face_crop_results, lip_dists
    
def _prepare_batch_input(_b_st=None, _mc=None, _ifs=None, _fcr=None, _apl=None, _acl=None, _obf=None,
                            _nlp=None, _nlc=None):
    _tif, _tofc = [], []
    
    _tcf, T_pose_landmarks, T_content_landmarks = [], [], [], []


        # 2.input face
    input_frame_idx = int(_ifs[mel_chunk_idx])
    face, coords = _fcr[input_frame_idx]
    _tcf.append(face)
    _tofc.append((face, coords))  ##input face
    # 3.pose landmarks
    T_pose_landmarks.append(_apl[input_frame_idx])
    T_content_landmarks.append(_acl[input_frame_idx])#EXPEIMENTAL BLOCK - FA Elimination

        # 3.background
    _tif.append(_obf[input_frame_idx].copy())
    
    # prepare pose landmarks
    T_pose = torch.zeros((self.T, 2, 74))  # 74 landmark
    for idx in range(self.T):
        T_pose_landmarks[idx] = sorted(T_pose_landmarks[idx],
                                        key=lambda land_tuple: self.ori_sequence_idx.index(land_tuple[0]))
        T_pose[idx, 0, :] = torch.FloatTensor(
            [T_pose_landmarks[idx][i][1] for i in range(len(T_pose_landmarks[idx]))])  # x
        T_pose[idx, 1, :] = torch.FloatTensor(
            [T_pose_landmarks[idx][i][2] for i in range(len(T_pose_landmarks[idx]))])  # y
        
        if idx == 2:
            T_content_landmarks[idx] = sorted(T_content_landmarks[idx],
                                        key=lambda land_tuple: self.ori_sequence_idx.index(land_tuple[0]))
            _al = T_content_landmarks[idx] + T_pose_landmarks[idx]
            _bl = self._blending_landmark_extraction(_al)
        
    T_pose = T_pose.unsqueeze(0)  # (1,T, 2,74)

    # landmark  generator inference
    Nl_pose, Nl_content = _nlp.cuda(), _nlc.cuda()  # (Nl,2,74)  (Nl,2,57)
    T_mels, T_pose = T_mels.cuda(), T_pose.cuda()
    with torch.no_grad():  # require    (1,T,1,hv,wv)(1,T,2,74)(1,T,2,57)
        _pc = self.lmkgen(T_mels, T_pose, Nl_pose, Nl_content)  # (1*T,2,57)
    T_pose = torch.cat([T_pose[i] for i in range(T_pose.size(0))], dim=0)  # (1*T,2,74)
    _tpfl = torch.cat([T_pose, _pc], dim=2).cpu().numpy()  # (1*T,2,131)

    return _tpfl, _tcf, _pc, _tofc, _tif, _bl #EXPERIMENTAL BLOCK - FA Elimination _bl added
#'''

def merge_face_contour_only(src_frame, generated_frame, face_region_coord, blending_landmarks): #function used in post-process
    """Merge the face from generated_frame into src_frame
    """
    input_img = src_frame
    _status = False
    y1, y2, x1, x2 = 0, 0, 0, 0
    if face_region_coord is not None:
        y1, y2, x1, x2 = face_region_coord
        input_img = src_frame[y1:y2, x1:x2]
        
    try:
        blending_landmarks[:,0] *= (x2 - x1)
        blending_landmarks[:,1] *= (y2 - y1)
        contour_pts = blending_landmarks + np.array([x1, y1])
        contour_pts = contour_pts.astype(int)
        ### 2) Make the landmark region mark image
        mask_img = np.zeros((src_frame.shape[0], src_frame.shape[1], 1), np.uint8)
        cv2.fillConvexPoly(mask_img, contour_pts, 255)
        ### 3) Do swap
        img = swap_masked_region(src_frame, generated_frame, mask=mask_img)

        _status = True
    except:
        _msg = 'ST03:Facial Landmark detection failed'
        logging.info(_msg)
        
    return img, _status
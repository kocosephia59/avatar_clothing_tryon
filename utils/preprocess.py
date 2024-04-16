import mediapipe as mp
import cv2
import os

def process_video(face_path):
    if not os.path.isfile(face_path):
        _msg = 'Input video file %s non-existent' % face_path
        logging.info(_msg)
        sys.exit(1)
    elif face_path.split('.')[1] in ['jpg', 'png', 'jpeg']: #if input a single image for testing
        ori_background_frames = [cv2.imread(face_path)]
    else:
        _msg = 'Reading video frames from %s' % face_path
        logging.info(_msg)

        video_stream = cv2.VideoCapture(face_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        ori_background_frames = []
        frame_idx = 0
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            ori_background_frames.append(frame)
            frame_idx = frame_idx + 1

    
    return ori_background_frames, fps
from PIL import Image
import numpy as np
import cv2

import cv2

import mediapipe as mp

def get_face_bounding_box(image_path):
    # Initialize MediaPipe Face Detection.
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Read the image with OpenCV.
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB (MediaPipe requirement).
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize face detection with a minimum confidence threshold.
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Process the image to detect faces.
        results = face_detection.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
                # Get the bounding box coordinates from the first detected face.
                bboxC = detection.location_data.relative_bounding_box
                img_height, img_width, _ = image.shape
                x_min = int(bboxC.xmin * img_width)
                y_min = int(bboxC.ymin * img_height) - 30
                box_width = int(bboxC.width * img_width)
                box_height = int(bboxC.height * img_height) + 30
                
                return (x_min, y_min, box_width, box_height)

    # Return None if no face is detected
    return None

def replace_faces(set1, set2):
    # Assuming set1 and set2 are lists of image file paths
    # and that both lists have the same length and corresponding images
    for img1_path, img2_path in zip(set1, set2):
        face_box1 = get_face_bounding_box(img1_path)
        face_box2 = get_face_bounding_box(img2_path)
        print (img2_path, img1_path)
        
        if face_box1 is not None and len(face_box1) > 0 and face_box2 is not None and len(face_box2) > 0:            # Load images with PIL for easier manipulation
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)

            # Crop the face from the first image
            face1 = img1.crop((face_box1[0], face_box1[1], face_box1[0] + face_box1[2], face_box1[1] + face_box1[3]))
            
            # Resize the cropped face to fit the second image's face bounding box size
            face1 = face1.resize((face_box2[2], face_box2[3]), Image.Resampling.LANCZOS)

            # Paste the resized face onto the second image
            img2.paste(face1, (face_box2[0], face_box2[1]))

            # Save or display the modified image
            img2.save('final/' + img2_path.split('/')[-1])



import os

def get_image_files(directory):
    # Define a tuple of acceptable image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    # List all files in the directory
    all_files = os.listdir(directory)
    # Filter files: check if the extension of each file is in the allowed image extensions
    image_files = [file for file in all_files if file.lower().endswith(image_extensions)]
    # Construct full file paths
    image_paths = [os.path.join(directory, file) for file in image_files]
    return image_paths


# Get image file sets
set2 = get_image_files('./output/tps_frames')
set1 = get_image_files('./input/square_target_frames')
set2 = sorted(set2)
set1 = sorted(set1)
#print("Set 1 images:", set1)
#print("Set 2 images:", set2)



# Get the combined images
combined_images = replace_faces(set1, set2)



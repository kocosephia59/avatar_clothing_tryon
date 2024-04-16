from PIL import Image
import numpy as np
import cv2

import cv2

def get_face_bounding_box(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale (improves face detection performance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load OpenCV's pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Return the coordinates of the first detected face (x, y, width, height)
    if len(faces) > 0:
        return faces[0]  # Returns the first face detected
    return None

def replace_faces(set1, set2):
    # Assuming set1 and set2 are lists of image file paths
    # and that both lists have the same length and corresponding images
    for img1_path, img2_path in zip(set1, set2):
        face_box1 = get_face_bounding_box(img1_path)
        face_box2 = get_face_bounding_box(img2_path)
        
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
set2 = get_image_files('/Users/yewongim/Projects/clothing_change/output')
set1 = get_image_files('/Users/yewongim/Projects/clothing_change/input/square_target_frames')

#print("Set 1 images:", set1)
#print("Set 2 images:", set2)



# Get the combined images
combined_images = replace_faces(set1, set2)

# Optionally, save or show the combined images
for index, image in enumerate(combined_images):
    image.save(f'./final/combined_image_{index:04}.jpg')

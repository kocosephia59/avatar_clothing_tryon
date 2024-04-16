import cv2
import numpy as np
import mediapipe as mp

from PIL import Image


# Initialize MediaPipe Face Detection.
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to get the bottom part of the bounding box.
def get_bottom_part_of_bounding_box(image):
    # Read the image.
    #image = cv2.imread(image_path)
    # Convert the image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect faces.
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            # Get the bounding box coordinates.
            bboxC = detection.location_data.relative_bounding_box
            img_height, img_width, _ = image.shape
            x_min = int(bboxC.xmin * img_width)
            y_min = int(bboxC.ymin * img_height)
            box_width = int(bboxC.width * img_width)
            box_height = int(bboxC.height * img_height)
            
            # Draw the bounding box on the image (optional).
            cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 255, 0), 2)

            # Calculate the bottom part of the bounding box.
            bottom_part = y_min + box_height
        return bottom_part


    else:
        return None

def remove_the_head_region (image_path, face_bottom, output_path):
    # Load the previously processed black and white image
    img = Image.open(image_path)
    img = img.convert("RGBA")  # Ensure image is in RGBA format
    data = img.getdata()

    new_data = []
    for y in range(img.height):
        for x in range(img.width):
            index = y * img.width + x
            if y < face_bottom:
                new_data.append((0, 0, 0, 255))  # Make the pixel black
            else:
                new_data.append(data[index])

    # Update image data and save
    img.putdata(new_data)
    img.save(output_path)

def invert_colors(image_path, output_path):
    # Open an existing image
    img = Image.open(image_path)
    img = img.convert("RGBA")  # Ensure image is in RGBA format

    # Convert image to an OpenCV format
    open_cv_image = np.array(img)
    # Convert RGB to BGR (OpenCV uses BGR)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Create a mask where white pixels are detected
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    debug = Image.fromarray(threshold)
    debug.save("threshold_img.png")
    

    # Inversion of the dilated mask to ensure white areas become black and vice versa
    # Apply dilation mask to image: where it's white, turn it black; else white
    inverted_dilation = cv2.bitwise_not(threshold)

    kernel_size = 75  # for example, 20 pixels padding
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(inverted_dilation, kernel, iterations=1)
    debug = Image.fromarray(dilated_mask)
    debug.save(output_path)
    

def create_mask(input_img_path, output_path):
    # Getting the bottom of the head frame and setting it white
    input_image = cv2.imread(input_img_path)
    bottom_img = get_bottom_part_of_bounding_box(input_image)
    invert_colors(input_img_path, "./tmp/test_crop_background.png")
    remove_the_head_region ("./tmp/test_crop_background.png", bottom_img, output_path)
    
    return bottom_img

input_img_path = "/Users/yewongim/Projects/clothing_change/input/cropped_output_0001.jpg"
create_mask(input_img_path, "mask_test_final.png")

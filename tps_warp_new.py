import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt
from utils.create_clothing_mask import create_mask

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
pose_img = mp_pose.Pose(static_image_mode=True, model_complexity=1, smooth_landmarks=True)

def plot_and_save_keypoints_with_results(img1, img2, results1, results2, output_path):
    # Convert images to RGB for consistent plotting
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Prepare the plot
    plt.figure(figsize=(10, 5))
    plt.imshow(img1_rgb)
    plt.imshow(img2_rgb, alpha=0.5)  # Make the second image semi-transparent

    # Plot keypoints from the first image results
    if results1.pose_landmarks:
        for landmark in results1.pose_landmarks.landmark:
            x = landmark.x * img1.shape[1]
            y = landmark.y * img1.shape[0]
            plt.plot(x, y, 'ro')  # Red circles for the first image

    # Plot keypoints from the second image results
    if results2.pose_landmarks:
        for landmark in results2.pose_landmarks.landmark:
            x = landmark.x * img2.shape[1]
            y = landmark.y * img2.shape[0]
            plt.plot(x, y, 'bo')  # Blue circles for the second image

    # Save the resulting plot to a file
    plt.axis('off')  # Hide the axis
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def apply_tps_warp(src_img, src_points, dst_points, mask):
    # Create grid to interpolate
    height, width = src_img.shape[:2]
    
    mask = cv2.resize(mask, (height, width), interpolation=cv2.INTER_NEAREST)

    grid_x, grid_y = np.meshgrid(np.arange(src_img.shape[1]), np.arange(src_img.shape[0]))
    # Flatten the grid
    flat_grid_x = grid_x.flatten()
    flat_grid_y = grid_y.flatten()
    
    # Extract coordinates where the mask is white (i.e., the mask value is greater than 0)
    mask_indices = np.where(mask > 0)
    masked_grid_x = grid_x[mask_indices]
    masked_grid_y = grid_y[mask_indices]
    
    # Compute TPS using RBF
    rbf_x = Rbf(dst_points[:, 0], dst_points[:, 1], src_points[:, 0], function='thin_plate', smooth=1)
    rbf_y = Rbf(dst_points[:, 0], dst_points[:, 1], src_points[:, 1], function='thin_plate', smooth=1)
    
    # Map the coordinates of the masked region using the computed TPS functions
    mapped_x = rbf_x(masked_grid_x, masked_grid_y)
    mapped_y = rbf_y(masked_grid_x, masked_grid_y)

    # Ensure mapped coordinates do not go outside image bounds
    mapped_x = np.clip(mapped_x, 0, width - 1)
    mapped_y = np.clip(mapped_y, 0, height - 1)

    # Initialize an output image
    output_image = np.copy(src_img)
    
    # Apply the TPS transformation only to the pixels within the masked region
    for i in range(src_img.shape[2]):  # Assuming image has multiple color channels
        # Use map_coordinates to interpolate the image at the new grid positions
        warped_channel = map_coordinates(src_img[:, :, i], [mapped_y, mapped_x], order=1)
        output_image[mask_indices[0], mask_indices[1], i] = warped_channel

    return output_image
  
    '''
    # Map the grid
    mapped_x = rbf_x(flat_grid_x, flat_grid_y).reshape(grid_x.shape)
    mapped_y = rbf_y(flat_grid_x, flat_grid_y).reshape(grid_y.shape)
    
    # Warp the image based on the grid mapping
    warped_image = cv2.remap(src_img, mapped_x.astype(np.float32), mapped_y.astype(np.float32), cv2.INTER_LINEAR)
    '''
    return warped_image

def animate_image_with_tps(image_path, video_path, output_video_path, mask):
    reference_image = cv2.imread(image_path)
    cap = cv2.VideoCapture(video_path)
    
    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    #out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_h, frame_w = reference_image.shape[:-1]
    _res = (frame_w, frame_h)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, _res)

    ref_landmarks = pose_img.process(reference_image) 
    frame_number = 0
    
    while cap.isOpened():
        
       

        ret, frame = cap.read()
        if not ret:
            break
        print ("Processing frame #", frame_number)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            
            # Extract keypoints
            frame_points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in results.pose_landmarks.landmark[9:]], dtype="float32")
            ref_points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in ref_landmarks.pose_landmarks.landmark[9:]], dtype="float32")
            
            # Apply TPS warp
            animated_image = apply_tps_warp(reference_image, ref_points, frame_points, mask)
           
            #plot_and_save_keypoints_with_results(frame_rgb, reference_image, results, ref_landmarks, f"landmark_debug_{frame_number}.png")
            frame_number += 1
            cv2.imwrite( f"./output/tps_frames/tps_warp_frame_{frame_number:04}.png", animated_image)
            out.write(animated_image)  # Write frame to video
        


    cap.release()
    out.release()
    cv2.destroyAllWindows()

target_img_path = './demo_img/red_dress.png'
mask_output_path = './tmp/inverted_mask_padded.png'
create_mask('/input/frames/cropped_output_0001.jpg', mask_output_path)
mask = cv2.imread(mask_output_path, 0)
animate_image_with_tps(target_img_path, 'output_video_square.mp4', 'tps_warp_video.mp4', mask)

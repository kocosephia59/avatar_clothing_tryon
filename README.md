# Avatar Clothing Change 

## Problem statement

Given an avatar make an application that can change the clothing of the avatar given a text prompt while keeping the identity the same. 

## Approaches

1. Grab a frame from the avatar reference video
2. Create a mask around the clothing region - to be implemented using huamn segmentation algorithm or simple edge detection such as Canny in conjunction with facelandmark detector (for the demo I used simple white background threshold however this relies on the avatar being on white background at all times)
3. Create an image of an outfit using diffusion model of choice (in my case I use Dalle with promt engineering)
4. Create an inpainted frame of the avatar using diffusion inpatinting(diffusion_inpainting.py)
5. Transfer original video's animation to the image of the avatar wearing a new outfit (tps_warp_new.py)
6. crop in the head of the original avatar to the motion transferred video and blend in (Have no been implemented yet)

## Running the Code

```bash
conda env create -f environment.yaml
conda activate tryon_clothing

```

## Needs to be implemented

* Adaptive mask : grab a new mask at each frame instead of fixed mask at base
* Head blending : Code had bug didn't have time to debug will see if I can find to get it up and running before the chat
* Debugging cv2.VideoWriter issue in tps_warp
* Need to integrated the codes together to make it a single run file.

Rough Flow is 

preprocess using preprocess.py
create target image using diffusion_inpainting.py
run tps_warp_new.py

Still need to write blend in the original head 

compline the images back to a video using 
'''bash
 ffmpeg -framerate 25 -i tps_warp_frame_%04d.png -c:v libx264 -r 25 -pix_fmt yuv420p output_video.mp4
 '''




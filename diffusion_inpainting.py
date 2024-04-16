from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import argparse

def get_different_clothing(prompt,input_img,input_mask,output_img):

    torch.backends.mkldnn.enabled = True
    torch.backends.quantized.enabled = True

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        #torch_dtype=torch.float16,
    )
    
    image = Image.open(input_img)
    mask_image = Image.open(input_mask)

    pipe.to("cpu")
    
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    image.save(output_img)

if __name__ == "__main__":
    '''
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Ar")

    # Add arguments
    parser.add_argument("--input", type=str, required=False, help="Path to the input file")
    parser.add_argument("--output", type=str, required=False, help="Path to the output file")
    parser.add_argument("--mode", type=str, default="default", help="Operation mode")

    # Parse the arguments
    args = parser.parse_args()

    '''
    
    prompt = "a women dressed as a superwomen"
    image_path = "input/cropped_output_0001.jpg"
    mask_image_path = "/Users/yewongim/Projects/clothing_change/inverted_mask.jpg"
    output_image_path = "./super_women.png"
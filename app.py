# Import required libraries 
import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

# Specify the path to the SAM checkpoint file
sam_checkpoint = "weights\sam_vit_h_4b8939.pth"

# Specify the SAM model type (e.g., 'vit_h')
model_type = 'vit_h'

# Specify the device to use ('cpu' or 'cuda')
device = 'cpu'

# Load the SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Create the Stable Diffusion Inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)

# List to store selected pixels for the mask
select_pixels = []

# Create a Gradio interface
with gr.Blocks() as demo:
    with gr.row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mask")
        output_img = gr.Image(label="Output")

    with gr.Blocks():
        prompt_text = gr.Textbox(lines=1, label='Prompt')

    with gr.Row():
        submit = gr.Button('Submit')

    def generate_mask(image, evt: gr.select_data):
        # Add the selected pixel coordinates to the list
        select_pixels.append(evt.index)

        # Set the image for the SAM predictor
        predictor.set_image(image)

        # Prepare the input points and labels for the mask prediction
        input_points = np.array(select_pixels)
        input_label = np.ones(input_points.shape[0])

        # Generate the mask using the SAM predictor
        mask, _, _ = predictor.predict(
            points_cords=input_points,
            point_labels=input_label,
        )

        # Convert the mask to PIL Image
        mask = Image.fromarray(mask[0, :, :])
        return mask

    def inpaint(image, mask, prompt):
        # Convert the images to PIL Image objects
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # Resize the images to the desired size (512x512)
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        # Perform inpainting using the Stable Diffusion Inpainting pipeline
        output = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
        ).images[0]

        return output

    # Configure the Gradio interface interactions
    input_img.select(generate_mask, [input_img], [mask_img])
    submit.click(
        inpaint,
        inputs=[input_img, mask_img, prompt_text],
        outputs=[output_img]
    )

if __name__ == "__main__":
    # Launch the Gradio interface
    demo.launch()

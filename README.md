# Stable Diffusion 2 - Segment Anything Model (SAM)

![image](https://github.com/prajwal-cn/GEN-AI---Stable-Diffusion-2---Segment-Anything-Model---SAM/assets/127007794/e3e692ec-fb3c-438d-bcce-e8e042499bf2)


## Introduction
Stable Diffusion 2 - Segment Anything (SAM) is a powerful model for segmenting objects in images. It utilizes stable diffusion, a deep learning technique that generates high-quality and coherent segmentations. This model can be used for various applications, such as image editing, object removal, image inpainting, and more.

![image](https://github.com/prajwal-cn/GEN-AI---Stable-Diffusion-2---Segment-Anything-Model---SAM/assets/127007794/e5658243-0b3a-404a-b889-ab974ab23d35)


## Installation
To use the Stable Diffusion 2 - Segment Anything (SAM) model, you need to install the required dependencies. You can install them using pip, preferably in a virtual environment. Here are the steps to follow:

1. Set up a virtual environment (optional but recommended):
   ```
   python -m venv sam-env
   source sam-env/bin/activate
   ```

2. Install the required dependencies:
   ```
   pip install torch torchvision diffusers transformers
   ```

3. Download the Stable Diffusion 2 - Segment Anything (SAM) model weights:
   ```python
   import torch

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = torch.hub.load("stabilityai/models", "sam_v2", device=device)
   ```

## Usage
Once you have installed the required dependencies and downloaded the model weights, you can use the Stable Diffusion 2 - Segment Anything (SAM) model to segment objects in images. Here's an example code snippet:

```python
import torch
from PIL import Image

# Load the image
image_path = "path/to/your/image.jpg"
image = Image.open(image_path).convert("RGB")

# Preprocess the image
preprocess = torch.hub.load("stabilityai/models", "sam_v2_preprocess")
input_tensor = preprocess(image).unsqueeze(0)

# Perform segmentation
output_tensor = model.sample(input_tensor)

# Post-process the output
postprocess = torch.hub.load("stabilityai/models", "sam_v2_postprocess")
segmentation = postprocess(output_tensor)

# Display the segmented image
segmentation_image = Image.fromarray(segmentation)
segmentation_image.show()
```

Make sure to replace `"path/to/your/image.jpg"` with the actual path to the image you want to segment.

## References
For more information on stable diffusion and the Stable Diffusion 2 - Segment Anything (SAM) model, refer to the following resources:

- Stable Diffusion 2 - Segment Anything (SAM) GitHub repository: [https://github.com/stabilityai/models](https://github.com/stabilityai/models)
- Stable Diffusion 2 - Segment Anything (SAM) model documentation: [https://diffusion-models.stabilityai.com/models/sam_v2](https://diffusion-models.stabilityai.com/models/sam_v2)

## License
The Stable Diffusion 2 - Segment Anything (SAM) model is licensed under the Apache License 2.0. Please refer to the LICENSE file in the model repository for more details.

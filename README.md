# Text-to-Video Diffusion Magic 🎬🎨

## Overview

This project utilizes Diffusers, a library designed for text-to-video generation, to create captivating videos based on textual prompts. By leveraging pre-trained models and cutting-edge techniques, this project enables users to effortlessly translate their imaginative descriptions into visually stunning animations.

## Getting Started

To begin using this project, follow these steps:

1. **Check your GPU**: Ensure that you have access to a GPU by running the provided code snippet.
2. **Install dependencies**: Install the required dependencies by executing the provided pip commands.
3. **Setup pipeline**: Set up the DiffusionPipeline with the desired configuration.
4. **Generate your video**: Input your desired prompts and generate the corresponding videos.
5. **Display the video**: Visualize the generated videos directly within the notebook.

## Instructions

### Step 1: Check GPU Availability

Before proceeding, verify that your environment has access to a GPU by running the provided code snippet.

```python
#@title Check your GPU!
!nvidia-smi -L
```

### Step 2: Install Dependencies

Install the necessary dependencies, including PyTorch, Diffusers, Transformers, Accelerate, and ImageIO.

```python
#@title Install dependencies
!pip -q install torch==2.0.0 git+https://github.com/huggingface/diffusers transformers accelerate imageio[ffmpeg]
```

### Step 3: Setup Pipeline

Initialize the DiffusionPipeline with the desired configuration settings, including model variant and precision.

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from IPython.display import HTML
from base64 import b64encode

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

pipe.components.keys()

pipe.components['tokenizer']

pipe.components['text_encoder']

pipe.components['unet']

```

### Step 4: Generate your video
```
prompt = 'A Teddy bear running in New York City' #@param {type:"string"}
negative_prompt = 'low quality' #@param {type:"string"}
video_duration_seconds = 3 #@param {type:"integer"}
num_frames = video_duration_seconds * 10
video_frames = pipe(prompt, negative_prompt="low quality", num_inference_steps=25, num_frames=num_frames).frames
video_path = export_to_video(video_frames)
```

### Step 5: Display the video
```
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML

def display_video(video):
    fig = plt.figure(figsize=(4.2,4.2))  #Display size specification
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    mov = []
    for i in range(len(video)):  #Append videos one by one to mov
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])

    #Animation creation
    anime = animation.ArtistAnimation(fig, mov, interval=100, repeat_delay=1000)

    plt.close()
    return anime
video = imageio.mimread(video_path)  #Loading video
HTML(display_video(video).to_html5_video())  #Inline video display in HTML5
```

## Result Videos

You can view the generated videos below:

- [Video 1](result_videos/video1.mp4)
- [Video 2](result_videos/video2.mp4)
- [Video 3](result_videos/video3.mp4)


## Usage

This project offers a convenient interface for generating videos from textual prompts. Users can experiment with various prompts to create diverse and engaging content.

## Example Prompts

Here are some example prompts to get you started:

1. "A Teddy bear running in New York City"
2. "A panda eating bamboo on a rock"
3. "A drone flythrough of a tropical jungle covered in snow"

Feel free to customize the prompts to suit your creative vision!

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to **Sam Whitteven** for providing the video assets used in this project.

Happy video creation! 🎬🎉

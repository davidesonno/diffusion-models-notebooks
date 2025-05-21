import json
import io
import zipfile

from PIL import Image
from google.colab import files

def read_prompts_json(file:str='prompts.json'):
    with open("prompts.json", "r", encoding="utf-8") as file:
        prompts = json.load(file)

    return prompts

def load_generative_model_pipeline(model_id:str, torch_dtype=None, device:str=None):
    # if model == "playgroundai/playground-v2.5-1024px-aesthetic":
    #     import os
    #     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch # importing here because the above code needs to be executed first
    from diffusers import (
        DiffusionPipeline,
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        StableDiffusionXLPipeline,
        KDPM2AncestralDiscreteScheduler,
        AutoencoderKL
    )
    if torch_dtype is None:
        torch_dtype = torch.float16
    pipe = None

    if model_id == "sd-legacy/stable-diffusion-v1-5":
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype) # TODO: check if float32 works
    
    elif model_id == "stabilityai/stable-diffusion-2-1":
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype) # TODO: check if float32 works
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
    elif model_id == "playgroundai/playground-v2.5-1024px-aesthetic":
        if not isinstance(torch_dtype, torch.float16):
            import warnings
            warnings.warn(
                f"The `torch_type` value passed {torch_dtype} might have unexpected behaviour with the"
                f'DiffusionPipleine(..., variant="fp16"). Consider using torch.float16 for consistency.'
                )
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, variant="fp16")

    elif model_id == "dataautogpt3/ProteusV0.3":
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype)
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch_dtype)
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if device is not None:
        pipe = pipe.to(device)

    return pipe

def generate_images_with_pipeline(pipeline, prompts:dict, v:int=2):
    generated_images = {} # subject: { filename: PIL.Image}
    images_per_prompt = prompts['images_per_prompt']

    for subject in prompts['subjects']:
        generated_images[subject] = {}

        for v, variant in enumerate(prompts['prompts_variants'], start=1):
            # prepare the prompt
            prompt_text = prompts['prompts_head'] + ' ' + variant + ' ' + subject 
            if v>0: print(f'Generating: {prompt_text}')
            prompt_text += prompts['prompts_tail']
            negative_prompt_text = ", ".join(prompts['negative_prompts'])

            # generate the images. The pipeline returns a list
            images = pipeline(
                prompt = [ prompt_text ] * prompts['images_per_prompt'],
                negative_prompt = [ negative_prompt_text ] * prompts['images_per_prompt'],
                height = prompts['images_width'],
                width = prompts['images_height'],
                progress_bar = v>1,
                output_type="pil" 
            ).images

            for i, image in enumerate(images, start=1):
                # the number of digits depends on the number of images to be generated
                filename = f'{subject}_var{v}_{i:0{images_per_prompt}}.jpg' #v1-001, v1-002, ..., v2-001, ...
                generated_images[subject][filename] = image

    return generated_images

def create_zip_from_images_dict(generated_images:dict, filename='generated_images.zip', start_download=True):
    if not filename.endswith('.zip'):
        filename += '.zip'

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for subject, files in generated_images.items():
            folder_name = subject.replace(' ', '_')  # Ensure safe folder name
            for filename, img in files.items():
                img_path = f"{folder_name}/{filename}"
                img_bytes = io.BytesIO()
                img.convert("RGB").save(img_bytes, format="JPEG")
                zipf.writestr(img_path, img_bytes.getvalue())

    # save to disk
    with open(filename, "wb") as f:
        f.write(zip_buffer.getvalue())

    if start_download:
        files.download(filename)

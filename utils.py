import json
import io
import zipfile
import textwrap
import google.colab as clb
import matplotlib.pyplot as plt

from PIL import Image

def read_prompts_json(file:str):
    with open(file, "r", encoding="utf-8") as file:
        prompts = json.load(file)

    return prompts

def load_generative_model_pipeline(model_id:str, torch_dtype:str=None):
    # if model == "playgroundai/playground-v2.5-1024px-aesthetic":
    #     import os
    #     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch # importing here because the above code needs to be executed before import torch
    from diffusers import (
        DiffusionPipeline,
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        StableDiffusionXLPipeline,
        KDPM2AncestralDiscreteScheduler,
        AutoencoderKL
    )

    if torch_dtype is None:
        torch_dtype = "float16"
    torch_dtype = getattr(torch, torch_dtype)

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

def extract_images_from_zip(zip_path):
    image_dict = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_path in zip_ref.namelist():
            if file_path.lower().endswith((".jpg", ".jpeg", ".png")) and '/' in file_path:
                role, filename = file_path.split('/', 1)
                with zip_ref.open(file_path) as file:
                    image = Image.open(file).convert('RGB')
                    if role not in image_dict:
                        image_dict[role] = {}
                    image_dict[role][filename] = image
    return image_dict

def caption_images(image_dict, processor, model, conditional_captioning_text: str = None, device: str = "cuda"):
    import torch
    model.eval()
    caption_dict = {}
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    for subject, images in image_dict.items():
        filenames = list(images.keys())
        pil_images = list(images.values())

        if conditional_captioning_text:
            inputs = processor(pil_images, [conditional_captioning_text] * len(pil_images), return_tensors="pt", padding=True).to(device)
        else:
            inputs = processor(pil_images, return_tensors="pt", padding=True).to(device)

        outputs = model.generate(**inputs)
        captions = processor.batch_decode(outputs, skip_special_tokens=True)
        caption_dict[subject] = {filename: caption for filename, caption in zip(filenames, captions)}

    return caption_dict

# for florence-type models
def caption_images_florence(image_dict, model, processor, task_prompt: str | list, text_input=None):
    if isinstance(task_prompt, str):
        task_prompt = [task_prompt]  # Ensure task_prompt is a list

    results = {}

    for subject, image_set in image_dict.items():
        results[subject] = {}
        for image_name, image in image_set.items():
            results[subject][image_name] = {}

            for task in task_prompt:
                if text_input is None:
                    prompt = task
                else:
                    prompt = task + text_input

                inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda')
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"].cuda(),
                    pixel_values=inputs["pixel_values"].cuda(),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task=task,
                    image_size=(image.width, image.height)
                )

                results[subject][image_name][task] = parsed_answer[task]

    return results

def parse_tasks_labels(tasks_labels:dict):
    parsed = {}

    for subject, subject_dict in tasks_labels.items():
        parsed[subject] = {}
        for image_name, image_dict in subject_dict.items():
            parsed[subject][image_name] = {
                'gender' : None,
                'race': []
            }
            for task, task_caption in image_dict.items():
                c:str = task_caption.lower()
                if 'CAPTION' in task: # so that it works with <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>, <MIXED_CAPTION> and <MIXED_CAPTION_PLUS>
                    if any(word in c for word in ['woman', 'female', 'girl']):
                        parsed[subject][image_name]['gender'] = 'f'
                    elif any(word in c for word in ['man', 'male', 'boy']):
                        parsed[subject][image_name]['gender'] = 'm'
                if 'ANALYZE' in task:
                    a = [element.split(': ') for element in c.split(', ')]
                    for element in a:
                        try:
                            tag, value = element
                            if tag == 'race':
                                parsed[subject][image_name]['race'].extend(value.split(';'))
                                break
                        except: pass

    return parsed

def plot_image_dict(image_dict, labels=None, max_per_key=None):
    num_rows = len(image_dict)
    max_cols = max(len(images) for images in image_dict.values())
    ncols = min(max_cols, max_per_key) if max_per_key is not None else max_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=(ncols * 2.5, num_rows * 2.5))

    if num_rows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] if not isinstance(ax, list) else ax for ax in axes]

    for row_idx, (subject, images) in enumerate(image_dict.items()):
        image_items = list(images.items())
        if max_per_key is not None:
            image_items = image_items[:max_per_key]

        for col_idx in range(ncols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(image_items):
                name, img = image_items[col_idx]
                ax.imshow(img)
                title_text = str(labels[subject][name]) if labels is not None else name
                wrapped_title = "\n".join(textwrap.wrap(title_text, width=35))
                ax.set_title(wrapped_title, fontsize=8)
            ax.axis('off')

        axes[row_idx][0].set_ylabel(subject, fontsize=12, rotation=0, labelpad=40)

    plt.tight_layout()
    plt.show()

def create_zip_from_images_dict(generated_images:dict, filename='generated_images.zip', start_download=False):
    if not filename.endswith('.zip'):
        filename += '.zip'

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for subject, files in generated_images.items():
            folder_name = subject.replace(' ', '_')  # Ensure safe folder name
            for img_name, img in files.items():
                img_path = f"{folder_name}/{img_name}"
                img_bytes = io.BytesIO()
                img.convert("RGB").save(img_bytes, format="JPEG")
                zipf.writestr(img_path, img_bytes.getvalue())

    # save to disk
    with open(filename, "wb") as f:
        f.write(zip_buffer.getvalue())

    if start_download:
        clb.files.download(filename)

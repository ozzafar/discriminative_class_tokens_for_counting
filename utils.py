from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
import torch

import kornia

from config import RunConfig
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.pipelines import AutoPipelineForText2Image

# From timm.data.constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def transform_img_tensor(image, config):
    """
    Transforms an image based on the specified classifier input configurations.
    """
    # image = kornia.geometry.transform.resize(image, 224, interpolation="bicubic")
    image = kornia.geometry.transform.resize(image, 224)
    image = kornia.geometry.transform.center_crop(image, (224, 224))
    # image = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(image)
    return image


def prepare_counting_model(config: RunConfig):
    match config.counting_model_name:
        case "clip":
            from transformers import CLIPModel
            return CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
        case"clip-count":
            from clip_count.run import Model
            return Model.load_from_checkpoint("clip_count/clipcount_pretrained.ckpt", strict=False).cuda()
def prepare_clip(config: RunConfig):
    # TODO move clip version to config
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    return clip, processor

def prepare_stable(config: RunConfig):
    # Generative model
    pretrained_model_name_or_path = "stabilityai/sdxl-turbo"

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    pipe = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path).to(
            "cuda"
    )
    scheduler = pipe.scheduler
    del pipe
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    return unet, vae, text_encoder, scheduler, tokenizer


def save_progress(text_encoder, placeholder_token_id, accelerator, config, save_path):
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    learned_embeds_dict = {config.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)

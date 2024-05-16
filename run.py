import time
from math import sqrt

import pandas as pd
import torch

import os
from pathlib import Path
import torch.utils.checkpoint
import itertools

from PIL import Image
from accelerate import Accelerator
from diffusers import AutoPipelineForText2Image
from torch import device
from transformers import CLIPProcessor, CLIPModel, YolosForObjectDetection, YolosImageProcessor

import prompt_dataset
import utils
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from config import RunConfig
import pyrallis
import shutil


def train(config: RunConfig):
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    torch.autograd.set_detect_anomaly(True)

    classification_model = utils.prepare_classifier(config)
    # TODO move to prepare_clip
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

    train_start = time.time()

    exp_identifier = (
        f'{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}_v1"
    )

    #### Train ####
    print(f"Start experiment {exp_identifier}")

    class_name = f"{config.amount} {config.clazz}"
    print(f"Start training class token for {class_name}")
    img_dir_path = f"img/sdxl-turbo/{config.clazz}_{config.amount}_{config.seed}_{config.lr}_v1/train"
    if Path(img_dir_path).exists():
        shutil.rmtree(img_dir_path)
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    # Stable model
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32
    ).to(device)

    unet, vae, text_encoder, scheduler, tokenizer = pipeline.unet, pipeline.vae, pipeline.text_encoder, pipeline.scheduler, pipeline.tokenizer

    #  Extend tokenizer and add a discriminative token ###
    class_infer = int(class_name.split()[0])
    prompt_suffix = " ".join(class_name.lower().split("_"))

    ## Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(config.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {config.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    ## Get token ids for our placeholder and initializer token.
    # This code block will complain if initializer string is not a single token
    ## Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(config.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(config.placeholder_token)

    # we resize the token embeddings here to account for placeholder_token
    text_encoder.resize_token_embeddings(len(tokenizer))

    #  Initialise the newly added placeholder token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Define dataloades

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids
        texts = [example["instance_prompt"] for example in examples]
        batch = {
            "texts": texts,
            "input_ids": input_ids,
        }

        return batch

    train_dataset = prompt_dataset.PromptDataset(
        prompt_suffix=prompt_suffix,
        tokenizer=tokenizer,
        placeholder_token=config.placeholder_token,
        number_of_prompts=config.number_of_prompts,
        epoch_size=config.epoch_size,
    )

    train_batch_size = config.batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Define optimization

    ## Freeze vae and unet
    utils.freeze_params(vae.parameters())
    utils.freeze_params(unet.parameters())

    ## Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    utils.freeze_params(params_to_freeze)

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )
    criterion = torch.nn.MSELoss().cuda()

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    if config.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    classification_model = classification_model.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    global_step = 0
    total_loss = 0
    min_loss = 99999

    # Define token output dir
    token_dir_path = f"token/{class_name}"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    token_path = f"{token_dir_path}/{exp_identifier}_{class_name}"

    #### Training loop ####
    for epoch in range(config.num_train_epochs):
        print(f"Epoch {epoch}")
        generator = torch.Generator(
            device=config.device
        )  # Seed generator to create the inital latent noise
        generator.manual_seed(config.seed)
        for step, batch in enumerate(train_dataloader):
            # setting the generator here means we update the same images
            classification_loss = None
            with accelerator.accumulate(text_encoder):
                generator.manual_seed(config.seed)

                # generate image
                t1 = time.time()
                # generate image
                image = pipeline(prompt=batch['texts'][0],
                                 num_inference_steps=1,
                                 output_type="pt",
                                 height=config.height,
                                 width=config.width,
                                 generator=generator,
                                 guidance_scale=0.0
                                 ).images[0]
                print(f"SDXL took {(time.time() - t1) / 60} minutes")

                image = image.unsqueeze(0)
                image_out = image
                image = utils.transform_img_tensor(image, config).to(device)

                prompt = [class_name.split()[-1]]

                with torch.cuda.amp.autocast():
                    orig_output = classification_model.forward(image, prompt)

                output = torch.sum(orig_output[0] / config.scale)

                if classification_loss is None:
                    classification_loss = criterion(
                        output, torch.HalfTensor([class_infer]).cuda()
                    ) / torch.HalfTensor([1]).cuda()
                else:
                    classification_loss += criterion(
                        output, torch.HalfTensor([class_infer]).cuda()
                    ) / torch.HalfTensor([1]).cuda()

                text_inputs = processor(text=prompt, return_tensors="pt", padding=True).to(accelerator.device)
                inputs = {**text_inputs, "pixel_values": image}
                clip_output = (clip(**inputs)[0][0] / 100).cuda()
                clip_output = config._lambda * (1 - clip_output)

                classification_loss += clip_output

                total_loss += classification_loss.detach().item()

                # log
                txt = f"On epoch {epoch} \n"
                with torch.no_grad():
                    txt += f"{batch['texts']} \n"
                    txt += f"{output.item()=} \n"
                    txt += f"Loss: {classification_loss.detach().item()} \n"
                    txt += f"Clip-Count loss: {classification_loss.detach().item() - clip_output.detach().item()} \n"
                    txt += f"Clip loss: {clip_output.detach().item()}"
                    with open("run_log.txt", "a") as f:
                        print(txt, file=f)
                    print(txt)
                    utils.numpy_to_pil(
                        image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                    )[0].save(
                        f"{img_dir_path}/{epoch}_{class_name}_{classification_loss.detach().item()}.jpg",
                        "JPEG",
                    )

                    # counting prediction heatmap
                    pred_density = orig_output[0].detach().cpu().numpy()
                    pred_density = pred_density / pred_density.max()
                    pred_density_write = 1. - pred_density
                    pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
                    pred_density_write = pred_density_write / 255.
                    img = TF.resize(image.detach(), (384)).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    heatmap_pred = 0.33 * img + 0.67 * pred_density_write
                    heatmap_pred = heatmap_pred / heatmap_pred.max()
                    utils.numpy_to_pil(
                        heatmap_pred
                    )[0].save(
                        f"{img_dir_path}/{epoch}_{class_name}_{classification_loss.detach().item()}_heatmap.jpg",
                        "JPEG",
                    )

                torch.nn.utils.clip_grad_norm_(
                    text_encoder.get_input_embeddings().parameters(),
                    config.max_grad_norm,
                )

                accelerator.backward(classification_loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = (
                        text_encoder.module.get_input_embeddings().weight.grad
                    )
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad

                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = (
                        torch.arange(len(tokenizer)) != placeholder_token_id
                )
                grads.data[index_grads_to_zero, :] = grads.data[
                    index_grads_to_zero, :
                ].fill_(0)

                if epoch == step == 0:
                    img_path = f"{img_dir_path}/actual.jpg"
                    utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(img_path, "JPEG")
                # Checks if the accelerator has performed an optimization step behind the scenes\n",
                if step == config.epoch_size - 1:
                    if total_loss > 2 * min_loss:
                        print("!!!!training collapse, try different hp!!!!")
                        # epoch = config.num_train_epochs
                        # break
                    if total_loss < min_loss:
                        min_loss = total_loss
                        current_early_stopping = config.early_stopping
                        # Create the pipeline using the trained modules and save it.
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(
                                f"Saved the new discriminative class token pipeline of {class_name} to pipeline_{token_path}"
                            )
                            img_path = f"{img_dir_path}/optimized.jpg"
                            utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(img_path,"JPEG")
                            # pipeline.save_pretrained(f"pipeline_{token_path}") # TODO unwrap text encoder accelerator
                    else:
                        current_early_stopping -= 1
                    print(
                        f"{current_early_stopping} steps to stop, current best {min_loss}"
                    )

                    total_loss = 0
                    global_step += 1

                optimizer.step()
                optimizer.zero_grad()

        if current_early_stopping < 0:
            break

    print(f"End training time: {(time.time() - train_start)/60} minutes")

def trainv2(config: RunConfig):
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    torch.autograd.set_detect_anomaly(True)

    classification_model = utils.prepare_classifier(config)
    # TODO move to prepare_clip
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

    train_start = time.time()

    exp_identifier = (
        f'{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}_v2"
    )

    #### Train ####
    print(f"Start experiment {exp_identifier}")

    class_name = f"{config.amount} {config.clazz}"
    print(f"Start training class token for {class_name}")
    img_dir_path = f"img/sdxl-turbo/{config.clazz}_{config.amount}_{config.seed}_{config.lr}_v2/train"
    if Path(img_dir_path).exists():
        shutil.rmtree(img_dir_path)
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    # Stable model
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32
    ).to(device)

    unet, vae, text_encoder, scheduler, tokenizer = pipeline.unet, pipeline.vae, pipeline.text_encoder, pipeline.scheduler, pipeline.tokenizer

    #  Extend tokenizer and add a discriminative token ###
    class_infer = int(class_name.split()[0])
    prompt_suffix = " ".join(class_name.lower().split("_"))

    ## Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(config.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {config.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    ## Get token ids for our placeholder and initializer token.
    # This code block will complain if initializer string is not a single token
    ## Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(config.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(config.placeholder_token)

    # we resize the token embeddings here to account for placeholder_token
    text_encoder.resize_token_embeddings(len(tokenizer))

    #  Initialise the newly added placeholder token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Define dataloades

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids
        texts = [example["instance_prompt"] for example in examples]
        batch = {
            "texts": texts,
            "input_ids": input_ids,
        }

        return batch

    train_dataset = prompt_dataset.PromptDataset(
        prompt_suffix=prompt_suffix,
        tokenizer=tokenizer,
        placeholder_token=config.placeholder_token,
        number_of_prompts=config.number_of_prompts,
        epoch_size=config.epoch_size,
    )

    train_batch_size = config.batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Define optimization

    ## Freeze vae and unet
    utils.freeze_params(vae.parameters())
    utils.freeze_params(unet.parameters())

    ## Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    utils.freeze_params(params_to_freeze)

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )
    criterion = torch.nn.MSELoss().cuda()

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    if config.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    classification_model = classification_model.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    global_step = 0
    total_loss = 0
    min_loss = 99999

    # Define token output dir
    token_dir_path = f"token/{class_name}"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    token_path = f"{token_dir_path}/{exp_identifier}_{class_name}"

    #### Training loop ####
    for epoch in range(config.num_train_epochs):
        print(f"Epoch {epoch}")
        generator = torch.Generator(
            device=config.device
        )  # Seed generator to create the inital latent noise
        generator.manual_seed(config.seed)
        for step, batch in enumerate(train_dataloader):
            step_start = time.time()
            # setting the generator here means we update the same images
            classification_loss = None
            with accelerator.accumulate(text_encoder):
                generator.manual_seed(config.seed)

                # generate image
                image = pipeline(prompt=batch['texts'][0],
                                 num_inference_steps=1,
                                 height=config.height,
                                 width=config.width,
                                 generator=generator,
                                 guidance_scale=0.0
                                 ).images[0]

                image = image.unsqueeze(0)
                image_out = image
                image = utils.transform_img_tensor(image, config).to(device)

                prompt = [class_name.split()[-1]]

                with torch.cuda.amp.autocast():
                    orig_output = classification_model.forward(image, prompt)

                pred_density1 = orig_output[0]
                pred_density1 = pred_density1 / pred_density1.max()
                mask = torch.sigmoid(100 * (pred_density1.unsqueeze(0) - 0.2))
                mask_max = F.max_pool2d(mask, kernel_size=2, stride=2)
                mask_max = mask_max.squeeze()

                dfs_iterative(mask_max)

                output = mask_max.sum()

                if classification_loss is None:
                    classification_loss = criterion(
                        output, torch.HalfTensor([class_infer]).cuda()
                    ) / torch.HalfTensor([1]).cuda()
                else:
                    classification_loss += criterion(
                        output, torch.HalfTensor([class_infer]).cuda()
                    ) / torch.HalfTensor([1]).cuda()

                text_inputs = processor(text=prompt, return_tensors="pt", padding=True).to(accelerator.device)
                inputs = {**text_inputs, "pixel_values": image}
                clip_output = (clip(**inputs)[0][0] / 100).cuda()
                clip_output = config._lambda * (1 - clip_output)

                classification_loss += clip_output

                total_loss += classification_loss.detach().item()

                # log
                txt = f"On epoch {epoch} \n"
                with torch.no_grad():
                    txt += f"{batch['texts']} \n"
                    txt += f"{output.item()=} \n"
                    txt += f"Loss: {classification_loss.detach().item()} \n"
                    txt += f"Clip-Count loss: {classification_loss.detach().item() - clip_output.detach().item()} \n"
                    txt += f"Clip loss: {clip_output.detach().item()}"
                    with open("run_log.txt", "a") as f:
                        print(txt, file=f)
                    print(txt)
                    utils.numpy_to_pil(
                        image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                    )[0].save(
                        f"{img_dir_path}/{epoch}_{class_name}_{classification_loss.detach().item()}.jpg",
                        "JPEG",
                    )

                    # counting prediction heatmap
                    pred_density = orig_output[0].detach().cpu().numpy()
                    pred_density = pred_density / pred_density.max()
                    pred_density_write = 1. - pred_density
                    pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
                    pred_density_write = pred_density_write / 255.
                    img = TF.resize(image.detach(), (384)).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    heatmap_pred = 0 * img + 1 * pred_density_write
                    heatmap_pred = heatmap_pred / heatmap_pred.max()
                    utils.numpy_to_pil(
                        heatmap_pred
                    )[0].save(
                        f"{img_dir_path}/{epoch}_{class_name}_{classification_loss.detach().item()}_heatmap.jpg",
                        "JPEG",
                    )

                torch.nn.utils.clip_grad_norm_(
                    text_encoder.get_input_embeddings().parameters(),
                    config.max_grad_norm,
                )

                accelerator.backward(classification_loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = (
                        text_encoder.module.get_input_embeddings().weight.grad
                    )
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad

                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = (
                        torch.arange(len(tokenizer)) != placeholder_token_id
                )
                grads.data[index_grads_to_zero, :] = grads.data[
                    index_grads_to_zero, :
                ].fill_(0)

                if epoch == step == 0:
                    img_path = f"{img_dir_path}/actual.jpg"
                    utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(img_path, "JPEG")
                # Checks if the accelerator has performed an optimization step behind the scenes\n",
                if step == config.epoch_size - 1:
                    if total_loss > 2 * min_loss:
                        print("!!!!training collapse, try different hp!!!!")
                        # epoch = config.num_train_epochs
                        # break
                    if total_loss < min_loss:
                        min_loss = total_loss
                        current_early_stopping = config.early_stopping
                        # Create the pipeline using the trained modules and save it.
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(
                                f"Saved the new discriminative class token pipeline of {class_name} to pipeline_{token_path}"
                            )
                            img_path = f"{img_dir_path}/optimized.jpg"
                            utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(img_path,"JPEG")
                            # pipeline.save_pretrained(f"pipeline_{token_path}") # TODO unwrap text encoder accelerator
                    else:
                        current_early_stopping -= 1
                    print(
                        f"{current_early_stopping} steps to stop, current best {min_loss}"
                    )

                    total_loss = 0
                    global_step += 1

                optimizer.step()
                optimizer.zero_grad()

        print(f"End step duration: {(time.time() - step_start) / 60} minutes")
        if current_early_stopping < 0:
            break

    print(f"End train time: {(time.time() - train_start) / 60} minutes")

def evaluate(config: RunConfig):

    classification_model = utils.prepare_classifier(config)

    class_name = "7 oranges"

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}_{config.num_of_SD_backpropagation_steps}"
    )

    # Stable model
    token_dir_path = f"token/{class_name}"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    pipe_path = f"pipeline_{token_dir_path}/{exp_identifier}_{class_name}"
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(config.device)

    tokens_to_try = [config.placeholder_token]
    # Create eval dir
    img_dir_path = f"img/{class_name}/eval"
    if Path(img_dir_path).exists():
        print("Img path exists {img_dir_path}")
        if config.skip_exists:
            print("baseline exists - skip it. Set 'skip_exists' to False regenerate.")
        else:
            shutil.rmtree(img_dir_path)
            tokens_to_try.append(config.initializer_token)
    else:
        tokens_to_try.append(config.initializer_token)

    Path(img_dir_path).mkdir(parents=True, exist_ok=True)
    prompt_suffix = " ".join(class_name.lower().split("_"))

    for descriptive_token in tokens_to_try:
        correct = 0
        prompt = f"A photo of {descriptive_token} {prompt_suffix}"
        print(f"Evaluation for the prompt: {prompt}")

        for seed in range(config.test_size):
            if descriptive_token == config.initializer_token:
                img_id = f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
                if os.path.exists(f"{img_id}_correct.jpg") or os.path.exists(
                    f"{img_id}_wrong.jpg"
                ):
                    print(f"Image exists {img_id} - skip generation")
                    if os.path.exists(f"{img_id}_correct.jpg"):
                        correct += 1
                    continue
            generator = torch.Generator(
                device=config.device
            )  # Seed generator to create the inital latent noise
            generator.manual_seed(seed)
            image_out = pipe(prompt, output_type="pt", generator=generator)[0]
            image = utils.transform_img_tensor(image_out, config)

            output = classification_model(image).logits
            pred_class = torch.argmax(output).item()

            if descriptive_token == config.initializer_token:
                img_path = (
                    f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
                    f"_{'correct' if pred_class == config.class_index else 'wrong'}.jpg"
                )
            else:
                img_path = (
                    f"{img_dir_path}/{seed}_{exp_identifier}_7 oranges.jpg"
                )

            utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[
                0
            ].save(img_path, "JPEG")

def evaluate_experiment(model, image_processor, image_path, clazz):
    count = 0
    image = Image.open(image_path)

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == clazz[:-1]:
            count += 1

    return count


def evaluate_experiments(config: RunConfig):
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    df = pd.DataFrame(columns=['class', 'seed', 'amount', 'sd_count', 'sd_optimized_count'])

    # detected_optimized_amount = evaluate_experiment(model, image_processor,  "img.png", "oranges")
    # Iterate over each subfolder inside the main folder
    for subfolder in os.listdir("img/sdxl-turbo"):

        version = "v2" if config.is_v2 else "v1"
        if version not in subfolder:
            continue

        if str(config.lr) not in subfolder:
            continue

        clazz, amount, seed, lr, v = subfolder.split('_')
        subfolder_path = os.path.join("img", subfolder, "train")

        detected_actual_amount = evaluate_experiment(model, image_processor, subfolder_path + "/actual.jpg", clazz)
        detected_optimized_amount = evaluate_experiment(model, image_processor, subfolder_path + "/optimized.jpg", clazz)

        new_row = {'class': clazz, 'seed': seed, 'amount': int(amount), 'sd_count': detected_actual_amount, 'sd_optimized_count': detected_optimized_amount}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    dir_name = "experiments"
    experiment_path = f"{dir_name}/experiment_{version}.pkl"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    df['sd_count_diff'] = abs(df['sd_count'] - df['amount'])
    df['sd_optimized_count_diff'] = abs(df['sd_optimized_count'] - df['amount'])

    df['sd_count_diff_norm'] = abs(df['sd_count'] - df['amount'])/df['amount']
    df['sd_optimized_count_diff_norm'] = abs(df['sd_optimized_count'] - df['amount'])/df['amount']

    df['sd_count_ratio'] = df[['sd_count', 'amount']].min(axis=1) / df[['sd_count', 'amount']].max(axis=1)
    df['sd_optimized_count_ratio'] = df[['sd_optimized_count', 'amount']].min(axis=1) / df[['sd_optimized_count', 'amount']].max(axis=1)

    df.to_pickle(experiment_path)

    # Calculate the results
    avg_sd_count_diff = df['sd_count_diff'].mean()
    avg_sd_optimized_count_diff = df['sd_optimized_count_diff'].mean()
    avg_diff_per_seed = df.groupby('seed').agg({'sd_count_diff': 'mean', 'sd_optimized_count_diff': 'mean'})
    avg_sd_count_diff_norm = df['sd_count_diff_norm'].mean()
    avg_sd_optimized_count_diff_norm = df['sd_optimized_count_diff_norm'].mean()
    avg_diff_per_seed_norm = df.groupby(['class','seed']).agg({'sd_count_diff_norm': 'mean', 'sd_optimized_count_diff_norm': 'mean'})

    print("\n*** Average Difference Results ***\n")
    print(f"SD average difference: {avg_sd_count_diff}, normalized: {avg_sd_count_diff_norm}")
    print(f"SD-optimized average difference: {avg_sd_optimized_count_diff}, normalized: {avg_sd_optimized_count_diff_norm}")
    print(f"differences per seed: {avg_diff_per_seed}")
    print(f"normalized differences per seed: {avg_diff_per_seed_norm}")

    print(f"\nSD MAE: {df['sd_count_diff'].mean()}, Ours MAE: {df['sd_optimized_count_diff'].mean()}")
    print(f"SD RMSE: {sqrt((df['sd_count_diff'] ** 2).mean())}, Ours RMSE: {sqrt((df['sd_optimized_count_diff'] ** 2).mean())}")
    print(f"SD MAE-N: {df['sd_count_diff_norm'].mean()}, Ours MAE-N: {df['sd_optimized_count_diff_norm'].mean()}")

def is_valid(matrix, row, col, visited):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    return (row >= 0 and row < num_rows and col >= 0 and col < num_cols and matrix[row][col] != 0 and not
    visited[row][col])

def dfs_rec(matrix, row, col, visited):
    if not visited[row][col]:
        matrix[row][col] = 0

    visited[row][col] = True

    steps = [0, 1, -1]
    # Perform DFS on the neighbors\n",
    for i in steps:
        for j in steps:
            if is_valid(matrix, row + i, col + j, visited):
                matrix[row + i][col + j] = 0
                dfs_rec(matrix, row + i, col + j, visited)

def dfs(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    visited = [[False] * num_cols for _ in range(num_rows)]

    # Traverse the matrix\n",
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j] > 0.9:
                visited[i][j] = True
                dfs_rec(matrix, i, j, visited)

def dfs_iterative(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    visited = [[False] * num_cols for _ in range(num_rows)]
    stack = []

    steps = [0, 1, -1]

    # Traverse the matrix
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j] > 0.9 and not visited[i][j]:
                visited[i][j] = True
                stack.append((i, j))

                while stack:
                    row, col = stack.pop()
                    if not visited[row][col]:
                        matrix[row][col] = 0
                    visited[row][col] = True

                    for x in steps:
                        for y in steps:
                            if is_valid(matrix, row + x, col + y, visited):
                                stack.append((row + x, col + y))

def run_experiments(config: RunConfig):
    classes = ["oranges","airplanes","cars","birds","cats","deers","dogs","frogs","horses","ships","trucks"]
    intervals = [(0, 5), (5, 10), (10, 15), (15, 30)]
    scales = [90, 80, 70, 60]
    seeds = [35]

    start = time.time()
    for clazz in classes:
        for i, interval in enumerate(intervals):
            scale = scales[i]
            for amount in range(interval[0] + 1, interval[1] + 1):
                for seed in seeds:
                    print(f"*** Running experiment {clazz=},{amount=},{seed=}")
                    config.clazz = clazz
                    config.scale = scale
                    config.amount = amount
                    config.seed = seed
                    try:
                        if config.is_v2:
                            trainv2(config)
                        else:
                            train(config)
                    except Exception as e:
                        print(f"train failed on {e}")

    print(f"Overall experiment time: {(time.time()-start)/3600} hours")

if __name__ == "__main__":

    config = pyrallis.parse(config_class=RunConfig)
    print(str(config).replace(" ",'\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check the arguments
    if config.train:
        if config.is_v2:
            trainv2(config)
        else:
            train(config)
    if config.evaluate:
        evaluate(config)
    if config.experiment:
        run_experiments(config)
    if config.evaluate_experiment:
        evaluate_experiments(config)

import time

import pandas as pd
import torch

import os
from pathlib import Path
import torch.utils.checkpoint
import itertools

from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from torch import device
from transformers import CLIPProcessor, CLIPModel, YolosForObjectDetection, YolosImageProcessor

import prompt_dataset
import utils
from InstaFlow.code.pipeline_rf import RectifiedFlowPipeline
import numpy as np
import cv2
import torchvision.transforms.functional as TF

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
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}"
    )

    #### Train ####
    print(f"Start experiment {exp_identifier}")

    class_name = f"{config.amount} {config.clazz}"
    print(f"Start training class token for {class_name}")
    img_dir_path = f"img/{config.clazz}_{config.amount}_{config.seed}/train"
    if Path(img_dir_path).exists():
        shutil.rmtree(img_dir_path)
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    # Stable model
    unet, vae, text_encoder, scheduler, tokenizer = utils.prepare_stable(config)

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
                pipeline = RectifiedFlowPipeline.from_pretrained(
                    "XCLIU/instaflow_0_9B_from_sd_1_5",
                    safety_checker=None,
                    torch_dtype=weight_dtype,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    scheduler=scheduler,
                ).to(device)

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

    print(f"End training time: {time.time() - train_start}")

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

        print(
            f"-----------------------Accuracy {descriptive_token} -----------------------------"
        )


def run_experiments(config: RunConfig):
    classes = ["oranges"]
    intervals = [(0, 5), (5, 10), (10, 15), (15, 30), (30, 50)]
    scales = [90, 80, 70, 60, 60]
    seeds = [35, 1]

    # classes = ["oranges"]
    # intervals = [(1, 2)]
    # scales = [90]
    # seeds = [35]

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
                    train(config)


def evaluate_experiment(model, image_processor, image_path, clazz):
    count = 0
    image = Image.open(image_path)

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == clazz[:-1]:
            count += 1

    return count


def evaluate_experiments(config: RunConfig):
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    df = pd.DataFrame(columns=['class', 'seed', 'amount', 'sd_count', 'sd_optimized_count'])

    # Iterate over each subfolder inside the main folder
    for subfolder in os.listdir("img"):
        if not subfolder.startswith(config.clazz):
            continue

        clazz, amount, seed = subfolder.split('_')
        subfolder_path = os.path.join("img", subfolder, "train")

        detected_actual_amount = evaluate_experiment(model, image_processor, subfolder_path + "/actual.jpg", clazz)
        detected_optimized_amount = evaluate_experiment(model, image_processor, subfolder_path + "/optimized.jpg", clazz)

        new_row = {'class': clazz, 'seed': seed, 'amount': amount, 'sd_count': detected_actual_amount, 'sd_optimized_count': detected_optimized_amount}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    dir_name = "experiments"
    experiment_path = f"{dir_name}/experiment_{time.strftime('%Y%m%d_%H%M%S')}.pkl"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    df.to_pickle(experiment_path)

    df = pd.read_pickle(experiment_path)


if __name__ == "__main__":

    config = pyrallis.parse(config_class=RunConfig)
    print(str(config).replace(" ",'\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check the arguments
    if config.train:
        train(config)
    if config.evaluate:
        evaluate(config)
    if config.experiment:
        run_experiments(config)
    if config.evaluate_experiment:
        evaluate_experiments(config)

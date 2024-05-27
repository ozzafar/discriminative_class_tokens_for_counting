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

from datasets.classes_datasets import yolo_classes, fsc147_classes
from diffusers.utils import load_image
from torchvision.transforms import transforms

from clip_count.run import Model
from clip_count.util import misc
from diffusers import AutoPipelineForText2Image, StableDiffusionXLControlNetPipeline, ControlNetModel
from torch import device
from transformers import YolosForObjectDetection, YolosImageProcessor, AutoModel, AutoProcessor, pipeline, \
    CLIPProcessor, CLIPModel

from datasets import prompt_dataset
import utils
import numpy as np
import cv2
import torchvision.transforms.functional as TF
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

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
    img_dir_path = f"img/{config.experiment_name}/{config.clazz}_{config.amount}_{config.seed}_{config.lr}_v1/train"
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
    criterion = torch.nn.L1Loss().cuda()

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
                    # txt += f"Clip-Count loss: {classification_loss.detach().item() - clip_output.detach().item()} \n"
                    # txt += f"Clip loss: {clip_output.detach().item()}"
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
                            img_path = f"{img_dir_path}/optimized.jpg"
                            utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(img_path,"JPEG")
                            accelerator.unwrap_model(
                                pipeline.text_encoder
                            ).save_pretrained(f"pipeline_{token_path}")
                            print(
                                f"Saved the new discriminative class token pipeline of {class_name} to pipeline_{token_path}"
                            )
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

    print(f"End training time: {(time.time() - train_start) / 60} minutes")

def evaluate(config: RunConfig):
    print("Evaluation - print image with discriminatory tokens, then one without.")
    # Stable model
    pipe_path = f"pipeline_token/{config.amount} {config.clazz}"
    pipe = AutoPipelineForText2Image.from_pretrained(
        pipe_path,
        torch_dtype=torch.float16
    ).to(device)

    print(f"{pipe_path=}")

    generator = torch.Generator(device=config.device)  # Seed generator to create the initial latent noise

    for descriptive_token in [config.placeholder_token, "some"]:
        generator.manual_seed(config.seed)
        prompt = f"A photo of {descriptive_token} {int(config.amount)} {config.clazz}"
        print(f"Evaluation for the prompt: {prompt}")

        with torch.no_grad():
            image_out = pipe(prompt=prompt,
                             num_inference_steps=1,
                             output_type="pt",
                             height=config.height,
                             width=config.width,
                             generator=generator,
                             guidance_scale=0.0
                             ).images[0]
            # image = utils.transform_img_tensor(image_out, config)

        img_dir_path = f"img/eval"
        Path(img_dir_path).mkdir(parents=True, exist_ok=True)

        utils.numpy_to_pil(
            image_out.unsqueeze(0).permute(0, 2, 3, 1).cpu().detach().numpy()
        )[0].save(
            f"{img_dir_path}/{prompt}.jpg",
            "JPEG",
        )

def yolo_evaluate_experiment(model, image_processor, image_path, clazz):
    count = 0
    image = Image.open(image_path)

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == clazz[:-1]:
            count += 1

    return count

def dino_evaluate_experiment(model, image_path, clazz):
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=clazz,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    return len(boxes)
def siglip_score(siglip_pipeline, image_path, amount, clazz):
    image = Image.open(image_path)

    outputs = siglip_pipeline(image, candidate_labels=[f"a photo of {amount} {clazz}"])
    score = round(outputs[0]["score"], 4)

    return score

def clip_score(model, processor, image_path, amount, clazz):
    image = Image.open(image_path)

    inputs = processor(text=[f"a photo of {amount} {clazz}"], images = image, return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)
    score = round(outputs[0][0].item()/100, 4)

    return score
def clipcount_evaluate_experiment(model, image_path, clazz):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize the image if necessary
        transforms.ToTensor()  # Convert the image to a tensor
    ])
    image = transform(image)

    with torch.cuda.amp.autocast():
        # print results
        raw_h, raw_w = image.size()[1:]

        patches, _ = misc.sliding_window(image, stride=128)
        # covert to batch
        patches = torch.from_numpy(patches).float().to(device)
        prompt = np.repeat(clazz, patches.shape[0], axis=0)
        output = model(patches, prompt)
        output.unsqueeze_(1)
        output = misc.window_composite(output, stride=128)
        output = output.squeeze(1)
        # crop to original width
        output = output[:, :, :raw_w]

        pred_cnt = torch.sum(output[0] / 70).item()

    return pred_cnt


def evaluate_experiments(config: RunConfig):
    yolo = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    yolo_image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    clipcount = Model.load_from_checkpoint("clipcount_pretrained.ckpt", strict=False).cuda()
    clipcount.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    dino = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                       "GroundingDINO/weights/groundingdino_swint_ogc.pth")

    df = pd.DataFrame(columns=['class', 'seed', 'amount', 'sd_count', 'sd_optimized_count', 'is_clipcount','is_yolo',
                               'sd_count2', 'sd_optimized_count2','actual_relevance_score','optimized_relevance_score',
                               'sd_count3', 'sd_optimized_count3'])

    # detected_optimized_amount = evaluate_experiment(model,  "img_7.png", "oranges")
    # Iterate over each subfolder inside the main folder
    folder = config.experiment_name
    for subfolder in os.listdir(f"img/{folder}"):

        version = "v2" if config.is_v2 else "v1"
        if version not in subfolder:
            continue

        if str(config.lr) not in subfolder:
            continue

        is_yolo, detected_actual_amount2, detected_optimized_amount2 = False, -1, -1
        clazz, amount, seed, lr, v = subfolder.split('_')
        subfolder_path = os.path.join("img", folder, subfolder, "train")
        is_clipcount = clazz in fsc147_classes

        print(f"evaluating {clazz=} {amount=}")

        path_actual = subfolder_path + "/actual.jpg" # for controlnet use something else
        path_optimized = subfolder_path + "/optimized.jpg"

        detected_actual_amount = clipcount_evaluate_experiment(clipcount, path_actual, clazz)
        detected_optimized_amount = clipcount_evaluate_experiment(clipcount, path_optimized, clazz)

        detected_actual_amount_dino = dino_evaluate_experiment(dino, path_actual, clazz)
        detected_optimized_amount_dino = dino_evaluate_experiment(dino, path_optimized, clazz)

        if clazz[:-1] in yolo.config.id2label.values():
            is_yolo = True
            detected_actual_amount2 = yolo_evaluate_experiment(yolo, yolo_image_processor, path_actual, clazz)
            detected_optimized_amount2 = yolo_evaluate_experiment(yolo, yolo_image_processor, path_optimized, clazz)

        actual_relevance_score = clip_score(clip, clip_processor, path_actual, amount, clazz)
        optimized_relevance_score = clip_score(clip, clip_processor, path_optimized, amount, clazz)

        new_row = {
            'class': clazz, 'seed': seed, 'amount': int(amount), 'sd_count': detected_actual_amount, 'sd_optimized_count': detected_optimized_amount,
            'is_clipcount' : is_clipcount, 'is_yolo' : is_yolo, 'sd_count2': detected_actual_amount2, 'sd_optimized_count2': detected_optimized_amount2,
            'actual_relevance_score': actual_relevance_score, 'optimized_relevance_score' :optimized_relevance_score,
            'sd_count3': detected_actual_amount_dino, 'sd_optimized_count3': detected_optimized_amount_dino
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    dir_name = "experiments"
    experiment_path = f"{dir_name}/experiment_{version}.pkl"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    df['sd_count_diff'] = abs(df['sd_count'] - df['amount'])
    df['sd_optimized_count_diff'] = abs(df['sd_optimized_count'] - df['amount'])
    df['sd_count_diff2'] = abs(df['sd_count2'] - df['amount'])
    df['sd_optimized_count_diff2'] = abs(df['sd_optimized_count2'] - df['amount'])
    df['sd_count_diff3'] = abs(df['sd_count3'] - df['amount'])
    df['sd_optimized_count_diff3'] = abs(df['sd_optimized_count3'] - df['amount'])

    df.to_pickle(experiment_path)

    print("\n*** Results ***\n")

    df=df[df['is_clipcount']==True]

    print(f"\nSD MAE (clipcount): {df[df['is_clipcount']==True]['sd_count_diff'].mean()}, Ours MAE: {df[df['is_clipcount']==True]['sd_optimized_count_diff'].mean()}")
    print(f"\nSD RMSE (clipcount): {sqrt((df[df['is_clipcount']==True]['sd_count_diff'] ** 2).mean())}, Ours RMSE: {sqrt((df[df['is_clipcount']==True]['sd_optimized_count_diff'] ** 2).mean())}")
    print(f"\nMAE (clipcount): {df[df['is_clipcount']==True].groupby('amount').agg({'sd_count_diff': 'mean', 'sd_optimized_count_diff': 'mean'})}")

    print(f"\nSD MAE (dino): {df[df['is_clipcount']==True]['sd_count_diff3'].mean()}, Ours MAE: {df[df['is_clipcount']==True]['sd_optimized_count_diff3'].mean()}")
    print(f"\nSD RMSE (dino): {sqrt((df[df['is_clipcount']==True]['sd_count_diff3'] ** 2).mean())}, Ours RMSE: {sqrt((df[df['is_clipcount']==True]['sd_optimized_count_diff3'] ** 2).mean())}")
    print(f"\nMAE (dino): {df[df['is_clipcount']==True].groupby('amount').agg({'sd_count_diff3': 'mean', 'sd_optimized_count_diff3': 'mean'})}")

    print(f"\nSD MAE (yolo): {df[df['is_yolo']==True]['sd_count_diff2'].mean()}, Ours MAE: {df[df['is_yolo']==True]['sd_optimized_count_diff2'].mean()}")
    print(f"\nSD RMSE (yolo): {sqrt((df[df['is_yolo']==True]['sd_count_diff2'] ** 2).mean())}, Ours RMSE: {sqrt((df[df['is_yolo']==True]['sd_optimized_count_diff2'] ** 2).mean())}")
    print(f"\nMAE (yolo): {df[df['is_yolo']==True].groupby('amount').agg({'sd_count_diff2':'mean','sd_optimized_count_diff2':'mean'})}")

    print(f"\nSD Relevance Score: {df[df['is_clipcount']==True]['actual_relevance_score'].mean()}, Ours Relevance Score: {df[df['is_clipcount']==True]['optimized_relevance_score'].mean()}")
    print(f"\nRelevance Score: {df[df['is_clipcount']==True].groupby('amount').agg({'actual_relevance_score':'mean','optimized_relevance_score':'mean'})}")

# def run_experiments(config: RunConfig):
#     experiments = [(3,"birds"),(5,"bowls"),(5,"chairs"),(5,"cups"),(10,"oranges"),(12,"cars"),(25,"grapes"),(25,"macroons"),(25,"pigeons"),(25,"see shells")]
#     for experiment in experiments:
#         amount, clazz = experiment[0],experiment[1]
#         config.amount=amount
#         config.clazz=clazz
#         train(config)

def run_controlnet(pipe, config):
    prompt = f"a realistic high resolution image of {config.amount} {config.clazz}"
    negative_prompt = "low quality, bad quality, sketches"

    print(f"Running ControlNet with prompt: {prompt}")

    # download an image
    image = load_image(
        f"controlnet/{config.amount}_dots.png"
    )

    # get canny image
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # generate image
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    image = pipe(
        prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image, height=512, width=512
    ).images[0]

    image.show()
    dir_name = f"img/controlnet/{config.clazz}_{config.amount}_{config.seed}_{config.lr}_v1/train"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    image.save(f"{dir_name}/optimized.jpg")

def run_experiments(config: RunConfig):
    # TODO stopped in calamari rings
    # classes = ["balls", "sea shells", "hot air balloons", "peppers", "bread rolls", "tomatoes", "geese", "seagulls",
    #            "peaches",
    #            "grapes", "watermelon", "beads", "candles", "oysters", "penguins", "strawberries", "pigeons",
    #            "macarons", "crows", "flamingos", "cranes", "boxes", "stamps", "watches", "bowls", "apples", "shoes",
    #            "windows", "cassettes", "ants", "birds", "books", "cups", "fishes", "people", "alcohol bottles",
    #            "bricks", "bottle caps", "plates", "comic books", "skateboard", "sheep", "buffaloes", "markers",
    #            "roof tiles", "pills", "keyboard keys", "carrom board pieces", "pencils", "coins", "boats", "elephants",
    #            "sunglasses", "cows", "pens", "stapler pins", "camels", "horses", "lipstick", "spoon", "bottles",
    #            "deers", "cement bags", "go game", "oranges", "cans", "chairs", "caps", "shirts", "jeans", "mini blinds",
    #            "zebras", "naan bread", "nuts", "crab cakes", "bees", "coffee beans", "gemstones", "cashew nuts", "buns",
    #            "kidney beans", "crayons", "matches", "bullets", "finger foods", "clams", "cotton balls", "cupcake tray",
    #            "green peas", "onion rings", "polka dots", "instant noodles", "red beans", "m&m pieces",
    #            "baguette rolls", "chicken wings", "ice cream", "meat skewers", "kitchen towels", "jade stones",
    #            "toilet paper rolls", "candy pieces", "spring rolls", "chewing gum pieces", "pearls", "donuts tray",
    #            "cupcakes", "lighters", "stairs", "shallots", "potatoes", "screws", "goldfish snack", "marbles",
    #            "polka dot tiles", "rice bags", "oyster shells", "mosaic tiles", "prawn crackers", "supermarket shelf",
    #            "sausages", "potato chips", "calamari rings", "biscuits", "croissants", "nails", "skis", "goats",
    #            "swans", "bananas", "kiwis", "tree logs", "eggs", "cars", "birthday candles", "sauce bottles", "cereals",
    #            "fresh cut", "milk cartons", "sticky notes", "nail polish", "cartridges", "legos", "flower pots",
    #            "flowers", "straws", "chopstick"]

    classes = fsc147_classes
    amounts = [5, 15, 25]
    seeds = [35]
    scale = 60

    print(f"{classes=}")

    if config.is_controlnet:
        # initialize the models and pipeline
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float32
        ).to(device)
        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float32
        ).to(device)
        pipe.enable_model_cpu_offload()

    start = time.time()
    for clazz in classes:
        for amount in amounts:
            for seed in seeds:
                print(f"*** Running experiment {clazz=},{amount=},{seed=}")
                config.clazz = (clazz+"s") if clazz in yolo_classes else clazz
                config.scale = scale
                config.amount = amount
                config.seed = seed
                try:
                    if config.is_controlnet:
                        run_controlnet(pipe, config)
                    else:
                        train(config)
                except Exception as e:
                    print(f"train failed on {e}")

    print(f"Overall experiment time: {(time.time() - start) / 3600} hours")

if __name__ == "__main__":

    config = pyrallis.parse(config_class=RunConfig)
    print(str(config).replace(" ", '\n'))

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

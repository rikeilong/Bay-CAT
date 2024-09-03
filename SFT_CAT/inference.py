import os
import sys
import argparse
import torch
from model.constants import IMAGE_TOKEN_INDEX,QUESTION_TOKEN_INDEX
from conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model, load_lora
from model.utils import disable_torch_init
from mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize,ToTensor
import numpy as np
import clip
import json
from tqdm import tqdm

torch.cuda.set_device(1)

def inference(model, image, audio, query, tokenizer):
    # conv = conv_templates["v1"].copy()
    conv = conv_templates["llama_2"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, QUESTION_TOKEN_INDEX, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # images=image[None,].cuda()
    # audios=audio[None,].cuda()
    images=image.cuda()
    audios=audio.cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            audios=audios,
            # do_sample=True,
            # temperature=0.05,
            do_sample=False,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/home/qilang/PythonProjects/Video-LLaMA-main/llama-2-7b-chat-hf/")
    parser.add_argument("--pretrain_CA", type=str, default="/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt-AVQA-adapter/CA/CA.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_v", type=str, default="/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt-AVQA-adapter/mm_projector_v/mm_projector_v.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_a", type=str, default="/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt-AVQA-adapter/mm_projector_a/mm_projector_a.bin")


    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args)
    model = model.cuda().to(torch.bfloat16)


    transform = Compose([
        ToTensor(),
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    images_feature = torch.from_numpy(np.load('/mnt/sda/imagebind_v_feats/{}.npy'.format('00001124'))).unsqueeze(0).to(torch.bfloat16)
    audios_feature = torch.from_numpy(np.load('/mnt/sda/imagebind_a_feats/{}.mp3.npy'.format('00001124'))).unsqueeze(0).to(torch.bfloat16)
    query = "<Q>How many instruments are sounding in the video?<Q>"

    inference_log = []

    print("query: ", query)
    print("answer: ", inference(model, images_feature,audios_feature, query+"\n<video>" , tokenizer))
    # with open(output_fn, "w") as f:
        # json.dump(inference_log, f, indent=4)
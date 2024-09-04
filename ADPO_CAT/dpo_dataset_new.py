import random
import copy
import json
import torch
import transformers
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from vtimellm import conversation as conversation_lib
# from vtimellm.mm_utils import tokenizer_image_token

from model.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import conversation as conversation_lib
from mm_utils import tokenizer_image_token

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@dataclass
class DataArguments:
    data_path: str = field(default='./dpo.json',
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    # feat_folder: Optional[str] = field(default='/home/qilang/Workshop/data/clip_vit_b32')


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt()) #将所有文本输入存进去

    # Tokenize conversations

    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0) #除开图像的token

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2) #
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) #把图像剪掉在计算长度
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX #将target全部遮罩

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids, #这里没有image的token，只有text的
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    return preprocess_llama_2(sources, tokenizer, has_image=has_image)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = json.load(open(data_path, "r"))
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(self.list_data_dict[i])

        try:
            # feature_path = '{}/{}.npy'.format(self.data_args.feat_folder, source['id'])
            feature_path_v = source['video']
            feature_path_a = source['audio']
            image = np.load(feature_path_v) 
            image = torch.from_numpy(image)
            audio = np.load(feature_path_a) 
            audio = torch.from_numpy(audio)
        except Exception as e:
            print(e)
            return random.choice(self)

        data_dict = preprocess(
            [source["conversations1"]],
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], 
                             labels=data_dict["labels"][0])
            
        data_dict2 = preprocess(
            [source["conversations2"]],
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict2 = dict(input_ids=data_dict2["input_ids"][0], 
                             labels=data_dict2["labels"][0])
        

        data_dict['image'] = image
        data_dict['audio'] = audio
        data_dict['chosen_input_ids'] = data_dict2['input_ids']
        data_dict['chosen_labels'] = data_dict2['labels']
        data_dict.update({'chosen':source['chosen']})
        data_dict.update({'rejected':source['rejected']})
        return data_dict
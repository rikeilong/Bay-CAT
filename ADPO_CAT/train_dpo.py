import os

import yaml
import json
import copy
import torch
import random
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from torch.utils.data import ConcatDataset
import transformers
from peft.peft_model import PeftModelForCausalLM
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments
from peft import PeftModel
import conversation as conversation_lib
from dpo_trainer.baseline_dpo_trainer import Dpo_Trainer
from model.mllmcatllm import MLLMCatLlamaForCausalLM
from model.builder import load_lora
from model.builder import load_pretrained_model, load_lora
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def float2bfloat(model):
    from utils_flash.llama_patch import upcast_layer_for_flash_attention
    torch_dtype = torch.bfloat16
    return(upcast_layer_for_flash_attention(model, torch_dtype))

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
        
    # hyper-parameters
    seed: Optional[int] = field(default=42, metadata={"help": "training and data seed."})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    
    # training parameters
    learning_rate: Optional[float] = field(default=4e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=0, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "maximum value of gradient norm"})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=True, metadata={"help": "whether to find unused parameters. set to False when `gradient_checkpointing` is False."}
    )
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[float] = field(default=1, metadata={"help": "number of trained eppchs."})
    logging_steps: Optional[int] = field(default=4, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=-1, metadata={"help": "the saving frequency"})
    evaluation_strategy: Optional[str] = field(default='no', metadata={"help": "the evaluation strategy"})
    eval_steps: Optional[float] = field(default=None, metadata={"help": "the evaluation frequency"})
    output_dir: Optional[str] = field(default="/home/qilang/PythonProjects/AVLLM/qformer_videochat/save_dir", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    run_name: Optional[str] = field(default="instructblip", metadata={"help": "name of the run"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    
    # lora parameters
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=2, metadata={"help": "the lora r parameter"})
    lora_target_modules: Optional[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"], metadata={"help": "the lora modules"})
    freeze_llm_proj: Optional[bool] = field(default=True, metadata={"help": "whether to freeze llama_proj module"})
    
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

@dataclass
class ModelArguments:
    # model_name_or_path: Optional[str] = field(default="/home/qilang/PythonProjects/vicuna-7b-v1.5/")
    model_name_or_path: Optional[str] = field(default="/home/qilang/PythonProjects/Video-LLaMA-main/llama-2-7b-chat-hf/")
    stage2_path: Optional[str] = field(default=None)
    # version: Optional[str] = field(default="v0")
    # version: Optional[str] = field(default="v1")
    version: Optional[str] = field(default="llama_2")
    # tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    pretrain_CA: Optional[str] = field(default=None) #after fine tune
    pretrain_mm_mlp_adapter_v: Optional[str] = field(default=None) #after feature alignment
    pretrain_mm_mlp_adapter_a: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_stage: int = field(default=1)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    # bits: int = field(
    #     default=16,
    #     metadata={"help": "How many bits to use."}
    # )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    # lora_enable: bool = False
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    # fp16: bool = True
    tf32: bool = True
    bf16: bool = True
    output_dir: str = field(default='./save_dir/')
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

@dataclass
class DataArguments:
    data_path: str = field(default='./dpo.json',
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    # feat_folder: Optional[str] = field(default='/home/qilang/Workshop/data/clip_vit_b32')

# callback used to save model
# !HACK: wandb upload failed!
class MyCallback(TrainerCallback):
    "A callback that prints a message at the end of training"
    def on_train_end(self, args, state, control, **kwargs):
        # save model
        if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
            print("Save model in the end of training")
            with open(os.path.join(args.output_dir, "training_args.yaml"), "w") as f:
                yaml.dump(args, f)
            # save lora weights
            if isinstance(kwargs['model'].base_model, PeftModelForCausalLM):
                kwargs['model'].base_model.save_pretrained(args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/home/qilang/PythonProjects/Video-LLaMA-main/llama-2-7b-chat-hf/")
    parser.add_argument("--pretrain_CA", type=str, default="/home/qilang/PythonProjects/MLLM_Cat/ckpt-adapter/CA/CA.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_v", type=str, default="/home/qilang/PythonProjects/MLLM_Cat/ckpt-AVQA-adapter/mm_projector_v/mm_projector_v.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_a", type=str, default="/home/qilang/PythonProjects/MLLM_Cat/ckpt-AVQA-adapter/mm_projector_a/mm_projector_a.bin")
    args = parser.parse_args()

    return args

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def save_model_trainer_projection(trainer: transformers.Trainer,output_dir: str):
    """Collects the state dict and dump to disk."""

    keys_to_match1 = ['CA']

    weight_to_save1 = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match1)
    trainer.model.config.save_pretrained(output_dir)

    CA_folder1 = os.path.join(output_dir, "CA")
    os.makedirs(CA_folder1, exist_ok=True)
    torch.save(weight_to_save1, os.path.join(CA_folder1, 'CA.bin'))

def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    parser2 = HfArgumentParser(ScriptArguments)
    script_args = parser2.parse_args_into_dataclasses()[0]

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    lora_config = AutoConfig.from_pretrained("/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt", trust_remote_code=True)
    model = MLLMCatLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=lora_config,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
    model.get_model().initialize_av_modules(model_args)

    for p in model.get_model().CA.parameters():
        p.requires_grad = True

    model = float2bfloat(model)
        

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]



    if training_args.bits in [4, 8]:
        model.get_model().CA.to(dtype=compute_dtype, device=training_args.device)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # tokenizer, model, context_len = load_pretrained_model(args)

    
    
    # build reference model
    lora_config2 = AutoConfig.from_pretrained("/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt", trust_remote_code=True)
    ref_model = MLLMCatLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=lora_config2,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )
    ref_model.config.use_cache = False
    # ref_model = PeftModel.from_pretrained(ref_model, "/home/qilang/PythonProjects/AVLLM/cross_Attn2/ckpt")
    # ref_model = ref_model.merge_and_unload()
    ref_model.get_model().initialize_av_modules(model_args)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        ref_model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        ref_model = prepare_model_for_kbit_training(ref_model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(ref_model, "enable_input_require_grads"):
            ref_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            ref_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

  
    if training_args.bits in [4, 8]:
        ref_model.get_model().CA.to(dtype=compute_dtype, device=training_args.device)


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in ref_model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    for n,p in ref_model.named_parameters():
        p.requires_grad = False
    
    from dpo_dataset_new import LazySupervisedDataset
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    
    # if not use gradient_checkpointing, do not set ddp_find_unused_parameters
    if not script_args.gradient_checkpointing:
        script_args.ddp_find_unused_parameters = False
    # if not use gradient_checkpointing, do not set ddp_find_unused_parameters
    if not script_args.gradient_checkpointing:
        script_args.ddp_find_unused_parameters = False
    
    # initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        # per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=script_args.ddp_find_unused_parameters,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        max_grad_norm=script_args.max_grad_norm,
        seed=script_args.seed,
    )
    
    # initialize the DPO trainer
    dpo_trainer = Dpo_Trainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )
    
    # model save callback
    dpo_trainer.add_callback(MyCallback())
    
    dpo_trainer.train()

    dpo_trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

    save_model_trainer_projection(trainer=dpo_trainer,output_dir='./save_dir/ckpt-adapter')
    
    # save script args
    with open(os.path.join(training_args.output_dir, "script_args.yaml"), "w") as f:
        yaml.dump(script_args, f)
    
if __name__ == "__main__":
    main()
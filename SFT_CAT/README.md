# Supervised Fine-tuning on CAT
CAT is based on LLaMA-2 7B and it includes three-stage training strategy. This step shows how to fine-tune the CAT.


* Download [ImageBind]((https://github.com/facebookresearch/ImageBind)) and [BLIP2](https://huggingface.co/docs/transformers/model_doc/blip-2) weights, and place them into the corresponding place.

## Get Started (Take Fine-tune on AVinstruct as example)

* 1. Place the instruction's json file in this script
```
-SFT_CAT
    -dataset.py
        -data_path: str = field(default='/AVinstruct/avqa_data1.json')
```

* 2. Set the output path, LoRA parameter settings, etc. in this script
```
-SFT_CAT
    -train.py
        -class TrainingArguments(transformers.TrainingArguments):
```

* 3. Start training in this script
```
-SFT_CAT
    -train_mem.py
```

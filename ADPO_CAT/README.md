# ADPO on CAT

## Get Started (Take dpo.json as example)

* 1. Load the fine-tuned parameters
```
-ADPO_CAT
    -train_dpo.py
      class ModelArguments:
          -pretrain_CA: Optional[str] = field(default=None) #after fine tune
          -pretrain_mm_mlp_adapter_v: Optional[str] = field(default=None) #after feature alignment
          -pretrain_mm_mlp_adapter_a: Optional[str] = field(default=None)
```

* 2. Place data that requires DPO in this script
```
-ADPO_CAT
    -dpo_dataset_new.py
      data_path: str = field(default='./dpo.json')
```

* 3. Start training in this script
```
-ADPO_CAT
    -train_dpo.py
```

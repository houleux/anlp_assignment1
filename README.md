# ASSIGNMENT-1

### Links:
1. HF link: [link](https://huggingface.co/iconically-mine/anlp-assignment1/tree/main)
2. GH link: [link](https://github.com/houleux/anlp_assignment1)
3. wandb link: [link](https://wandb.ai/rithvik-achutuni-iiit-hyderabad/anlp-assignment-1/workspace?nw=nwuserrithvikachutuni)

### Setup:
```
pip install pip install torch wandb huggingface_hub regex matplotlib
wandb login
```

### Simulate

#### Split dataset
```
python utils/split_dataset.pyt
```

#### Byte Packing
```
python utils/byte_packing.py
```

#### Pretrain tokenizers
```
python utils/train_tokenizers.py
```

#### Pretrain entropy model
```
python utils/pretrain_entropy.py
```


----
```
python src/train.py
``` 

Final command that trains all configs and also evaluates them
This repository outlines the method for fine-tuning and aligning a medical LLM with Llama 3.1 using LoRA and DPO.

The Meta-Llama-3.1-8B-Instruct model is first fine-tuned on medical dialogue data through LoRA, followed by alignment with medical preference data using DPO.

The fine-tuning process utilizes the LLaMA-Factory tool.

## 1 Install LLaMA-Factory

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[metrics]
```

## 2 Download llama3.1 model files  

```bash
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

 or 

```bash
git clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3.1-8B-Instruct.git
```

## 3 Finetune llama3.1 medical dialogue LLM using LoRA

dataset source：https://huggingface.co/datasets/shibing624/medical

This project uses the finetune dataset `train_en_1.json`.

### 3.1 Convert the dataset format

```bash
python convert_sft_dataset.py --in_file train_en_1.json --out_file medical_dialogue.json
```

Place `medical_dialogue.json` into the `LLaMA-Factory/data` folder.

Modify the file `dataset_info.json` in the `LLaMA-Factory/data` folder.

```json
  "alpaca_medical": {
    "file_name": "medical_dialogue.json"
  },
```

### 3.2  Finetune llama3.1 using LoRA

Modify  LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml and run

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

llama3_lora_sft.yaml
```yaml
### model
model_name_or_path: /your/folder/to/models/Meta-Llama-3.1-8B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj

### dataset
dataset: alpaca_medical
template: llama3
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /your/folder/to/models/llama3-lora-medical
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```



### 3.3  Merge LoRA adaper file 

Modify examples/merge_lora/llama3_lora_sft.yaml and run

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

llama3_lora_sft.yaml

```yaml
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters

### model
model_name_or_path: /your/folder/to/models/Meta-Llama-3.1-8B-Instruct
adapter_name_or_path: /your/folder/to/models/llama3-lora-medical
template: llama3
finetuning_type: lora

### export
export_dir: /your/folder/to/models/llama3-lora-medical-full
export_size: 5
export_device: cuda
export_legacy_format: false
```

<img src=".\images\training_loss_sft.png" alt="training_loss_sft" style="zoom:67%;" />

###  3.4 LLM Inference 

Modify examples/inference/llama3.yaml and run

```bash
llamafactory-cli chat examples/inference/llama3.yaml
```

llama3.yaml

```yaml
model_name_or_path: /your/folder/to/models/llama3-lora-medical-full
template: llama3
```
Here is an example of inference outputs from a chat with a medical LLM.

<img src=".\images\inference_example.png" alt="inference_example" style="zoom:67%;" />

## 4 Align medical LLM using DPO

dataset source：https://huggingface.co/datasets/shibing624/medical

This project uses the reward dataset `train.json`.

### 4.1 Convert the dataset format

```bash
python convert_dpo_dataset.py --in_file train.json --out_file medical_reward.json
```

Place `medical_reward.json` into the `LLaMA-Factory/data` folder.

Modify the file `dataset_info.json` in the `LLaMA-Factory/data` folder.

```json
  "dpo_medical": {
    "file_name": "medical_reward.json",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  },
```

### 4.2  Align finetuned llama3.1 medical LLM using DPO

Modify examples/train_lora/llama3_lora_dpo.yaml and run

```bash
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```


```yaml
### model
model_name_or_path: /your/folder/to/models/llama3-lora-medical-full

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: q_proj, v_proj
pref_beta: 0.1
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]

### dataset
dataset: dpo_medical
template: llama3
cutoff_len: 512 #1024
max_samples: 500 # 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /your/folder/to/models/llama3-lora-medical-dpo
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

 

<img src=".\images\training_loss_dpo.png" alt="training_loss_dpo" style="zoom:67%;" />

<img src=".\images\training_rewards_accuracies.png" alt="training_rewards_accuracies" style="zoom:67%;" />

























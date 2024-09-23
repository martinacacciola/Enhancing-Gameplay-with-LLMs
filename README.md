# Enhancing Minecraft Gameplay with LLMs
This project aims to enhance Minecraft game-play by integrating Large Language Models (LLMs) as in-game AI assistants. We compare the capabilities of various sized LLMs and proceed with the fine-tuning of smaller models on specific Minecraft knowledge.

# Large Language Model Training in Minecraft

This script is used to train a Fast Language Model using the UnSloth library. The model is trained on a dataset of Minecraft question-answer pairs.

## Dependencies

Before running the code, make sure to have the following dependencies installed:

- `unsloth`
- `torch`
- `datasets`
- `transformers`
- `trl`

## Model Configuration

The model can be loaded in 4-bit precision for faster computation. A list of 4-bit models available from UnSloth is provided in the script.

The model is further configured with PEFT (Progressive Embedding Factorization Transformer) parameters such as `r`, `target_modules`, `lora_alpha`, `lora_dropout`, `bias`, `use_gradient_checkpointing`, `random_state`, `use_rslora`, and `loftq_config`.

## Dataset

The dataset used is [minecraft-question-answer-700k](https://huggingface.co/datasets/naklecha/minecraft-question-answer-700k). The dataset has been formatted according to the `alpaca_prompt` format.

## Training

The model is trained using the `SFTTrainer` from the `trl` library. The training arguments are configured with parameters such as `per_device_train_batch_size`, `gradient_accumulation_steps`, `warmup_steps`, `max_steps`, `learning_rate`, `fp16`, `bf16`, `logging_steps`, `optim`, `weight_decay`, `lr_scheduler_type`, `seed`, and `output_dir`.

Please ensure to install all the necessary dependencies before running the script. Also, adjust the training parameters according to your computational resources and requirements. 

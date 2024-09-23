# Enhancing Minecraft Gameplay with LLMs
This project aims to enhance Minecraft gameplay by integrating Large Language Models (LLMs) as in-game AI assistants. We compare the capabilities of various-sized LLMs and proceed with the fine-tuning of smaller models on specific Minecraft knowledge. The role of the AI assistant is to provide real-time guidance, and strategy suggestions, contribute creative ideas and enhance the overall gaming experience. Our methodology involves embedding various-sized LLMs into the game environment and systematically analysing their impact on gameplay through quantitative metrics and qualitative user feedback. By comparing the efficiency, responsiveness, and creativity of each model, we identify strengths and limitations inherent to both state-of-the-art and local LLMs. In addition to this goal, we fine-tune a 2-billion parameters Gemma model, trained on the content of the Minecraft Wiki. This will equip the model with extensive knowledge about the game, making it an in-game expert. The findings of this study offer insights into the practical applications of LLMs in interactive environments and inform future developments in AI-assisted gaming experiences. Our comparative analysis revealed that GPT-4 turbo outperformed other models in terms of compliance, enjoyment, and believability, making it the most suitable model for the Minecraft environment.

<img width="349" alt="image" src="https://github.com/user-attachments/assets/ddaec680-0f63-485b-94ac-141082e2135d">

# Gemma 2b Training
The script `gemma2b_training.py` is used to train a Fast Language Model using the UnSloth library. The model is trained on a dataset of Minecraft question-answer pairs.
<img width="560" alt="image" src="https://github.com/user-attachments/assets/3f596f87-3e52-46f4-af8b-c529dc7eb8d3">


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

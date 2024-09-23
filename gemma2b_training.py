from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

max_seq_length = 4096 
dtype = None
load_in_4bit = True

# MODELS, more here: https://huggingface.co/unsloth
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

alpaca_prompt = """Below is an instruction that describes a question, paired with an answer. Write a response that appropriately completes the request.

### Question:
{}

### Answer:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    questions = examples["question"]
    answers   = examples["answer"]
    texts = []
    for questions, answers in zip(questions, answers):
        text = alpaca_prompt.format(questions, answers) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


# DATASET
dataset = load_dataset("naklecha/minecraft-question-answer-700k", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")


# TRAINING
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,

    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
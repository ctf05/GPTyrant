from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from LoadModel import load_model
import os


def train_gpt2_model(text_file_path):

    #Function to train a GPT-2 model on a given text file.

    model_name = os.path.splitext(os.path.basename(text_file_path))[0]

    output_dir = f"Models/{model_name}"

    model, tokenizer = load_model("gpt2")

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_file_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained(output_dir)
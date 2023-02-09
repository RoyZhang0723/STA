# from datasets import load_dataset, GenerateMode
from datasets import load_dataset, DownloadMode
from transformers import AutoTokenizer, BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

class Dataset:
    train_dataloader = None
    eval_dataloader = None
    test_dataloader = None

def load_dataset_for_transformer(config):
    # data loader
    raw_datasets = load_dataset(config.dataset_script, download_mode=DownloadMode.FORCE_REDOWNLOAD)
    # raw_datasets = load_dataset(config.dataset_script)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize_function(example):
        # print('enter')
        # print(example)
        return tokenizer(example['text'], truncation=True, max_length=256)
        # return tokenizer(example['text'], truncation=True)
        # return tokenizer(example['text'])

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare for training
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["text"]
    )

    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # tokenized_datasets["train"].column_names

    # We can then check that the result only has columns that our model will accept:
    # ['attention_mask', 'input_ids', 'label', 'token_type_ids']

    # Now that this is done, we can easily define our dataloaders:
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        # tokenized_datasets["train"], shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        # tokenized_datasets["train"], shuffle = True, batch_size = config.train_batch_size, collate_fn = data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=config.test_batch_size, collate_fn=data_collator
    )

    ds = Dataset()
    ds.train_dataloader = train_dataloader
    ds.test_dataloader = test_dataloader
    return ds

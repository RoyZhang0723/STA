import logging
import torch
from datasets import load_metric
from tqdm.auto import tqdm #进度条
import copy

def evaluation(model, config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    metric = load_metric("accuracy.py")

    model.eval()

    # result recoding
    golds = []
    preds = []
    # result recoding

    logging.info('###evaluating...')
    # progress_bar_eval = tqdm(range(len(config.eval_dataloader)))
    for batch in config.eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

        # progress_bar_eval.update(1)

        # result recoding
        golds.extend(batch['labels'].cpu().numpy())
        preds.extend(predictions.cpu().numpy())
        # result recoding

    logging.info(metric.compute())

    # result recoding
    return golds, preds
    # result recoding

from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler

from evaluate import *


def training(config):
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels)

    optimizer = AdamW(model.parameters(), lr = config.learning_rate)

    num_epochs = config.epoch_num
    num_training_steps = num_epochs * len(config.train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    print(num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    i_step = 1

    model.train()
    for epoch in range(num_epochs):
        for batch in config.train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # evaluation():
            if i_step % config.eval_per_step == 0:

                golds, preds = evaluation(model, config)
                fplog = open('../output/' + config.model_name.split('/')[-1] + '_' + config.dataset_script.split('/')[-1].split('.')[0] + '_' + str(i_step) + \
                             '.txt', 'w', encoding='utf-8')
                for i in range(len(golds)):
                    gold = golds[i]
                    pred = preds[i]
                    fplog.writelines(str(gold) + ',' + str(pred) + '\n')
                    fplog.flush()

            i_step += 1

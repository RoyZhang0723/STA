class Config:
    def __init__(self):
        # hardware
        self.use_gpu = "0"

        self.dataset_script = 'dataloader_openrice_competition.py'

        self.model_name = 'bert-base-chinese'

        self.train_batch_size = 4
        self.eval_batch_size = 32
        self.test_batch_size = 32

        self.num_labels = 5

        # training
        self.epoch_num = 3
        self.learning_rate = 2e-5
        self.eval_per_step = 5000

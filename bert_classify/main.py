import data_reader
import config
import train
import random
import numpy as np
import torch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

args = config.ARGS
set_seed(args)

tokenizer = data_reader.load_tokenizer(config.ARGS)  
train_dataset = data_reader.load_and_cache_examples(config.ARGS, tokenizer, mode="train")
dev_dataset = data_reader.load_and_cache_examples(config.ARGS, tokenizer, mode="dev")
test_dataset = data_reader.load_and_cache_examples(config.ARGS, tokenizer, mode="test")

trainer = train.Trainer(args, train_dataset, dev_dataset, test_dataset)

trainer.train()
trainer.load_model()
trainer.evaluate("dev")
trainer.evaluate("test")


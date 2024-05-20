from transformers import BertConfig, BertTokenizer
from model.model import ClsBERT

MODEL_CLASSES = {
    'bert': (BertConfig, ClsBERT, BertTokenizer),
}

MODEL_PATH_MAP = {
    'bert': '/workspace/huggingface/bert-base-uncased', # path to bert base model
}

class Args():
    task =  None
    data_dir =  None
    intent_label_file =  None
    
ARGS = Args()
ARGS.task = "MetaQA-hop-predict" # name of task (key in data_reader.processors)
ARGS.data_dir = "" # base path of the data file
ARGS.intent_label_file = "intent_label.txt" # file with all category tags (one tag per line)
ARGS.seed = 12 # random seed
ARGS.max_seq_len = 24 # max length of each input text
ARGS.model_type = "bert" # model name (key in MODEL_PATH_MAP)
ARGS.model_dir = "" # output model dictionary
ARGS.query_file = 'query.txt' # path to the file of input data
ARGS.label_file = 'label.txt' # path to the file of the label of input data
ARGS.model_name_or_path = MODEL_PATH_MAP[ARGS.model_type]
ARGS.train_batch_size = 64 # train batchsize
ARGS.eval_batch_size = 64 # evaluate batchsize
ARGS.device = "" 
ARGS.dropout_rate = 0.1 # linear classfier drop_rate
ARGS.max_steps = 3001 # max steps to update parameters
ARGS.gradient_accumulation_steps = 1 # gradient accumulation steps
ARGS.num_train_epochs = 1 # max epoch to train
ARGS.logging_steps = 1000 # how many steps to log
ARGS.save_steps = 1000 # how many steps to save model
ARGS.adam_epsilon = 1e-8 # to avoid division by zero
ARGS.weight_decay = 1e-5 # weight decay
ARGS.learning_rate = 1e-5 # learning rate
ARGS.max_grad_norm = 1.0 # max grad norm
ARGS.warmup_steps = 100 # AdamW warm_up steps

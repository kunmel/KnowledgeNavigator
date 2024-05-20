import os
from model.model import ClsBERT
import data_reader
import logging
import utils
import torch

logger = logging.getLogger(__name__)
utils.init_logger()

def load_model(args):
    intent_label_list = data_reader.get_intent_labels(args)
    # Check whether model exists
    if not os.path.exists(args.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = ClsBERT.from_pretrained(args.model_dir,
                                        args=args,
                                        intent_label_list=intent_label_list)
        model.to(args.device)
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")
    return model

def load_classify_decoder(args):
    with open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8') as intent_labels:
        labels = intent_labels
        decoder = {}
        for idx, label in enumerate(labels):
            decoder[idx] = label.strip()
    return decoder
    
def predict(model, args, example, tokenizer):
    model.to(args.device)
    # Setting based on the current model type
    cls_token = tokenizer.cls_token   # [CLS]
    sep_token = tokenizer.sep_token   # [SEP]
    unk_token = tokenizer.unk_token   # [UNK]
    pad_token_id = tokenizer.pad_token_id  # [PAD] -> 0

    # Tokenize words
    tokens = []
    example = example.split(" ")
    for w in example: 
        toks = tokenizer.tokenize(w)
        tokens.extend(toks)

    # Account for [CLS] and [SEP] 
    special_tokens_count = 2
    if len(tokens) > args.max_seq_len - special_tokens_count:
        tokens = tokens[:(args.max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]
    token_type_ids = [0] * len(tokens)

    # Add [CLS] token
    tokens = [cls_token] + tokens
    token_type_ids = [0] + token_type_ids
    
    # token 2 id
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. 
    attention_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = args.max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length) 
    
    
    with torch.no_grad():
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=args.device)
        input_ids = torch.unsqueeze(input_ids, 0)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=args.device)
        token_type_ids = torch.unsqueeze(token_type_ids, 0)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=args.device)
        attention_mask = torch.unsqueeze(attention_mask, 0)
        outputs = model.predict(input_ids, attention_mask, token_type_ids)
        intent_logits = outputs[0]
        intent_preds = intent_logits.detach().cpu().numpy()
        pred_result = list(intent_preds[0])
        return pred_result.index(max(pred_result))
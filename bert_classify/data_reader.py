import os
import copy
import json
import logging
import torch
import config
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]

class InputExample(object):
    def __init__(self, guid, words, intent_label=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):    
        output = copy.deepcopy(self.__dict__)
        print('-------')
        print(output)
        print('-------')
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class ClsProcessor(object):
    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)      
        self.input_text_file = args.query_file  
        self.intent_label_file = args.label_file 
    
    @classmethod  
    def _read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines  

    def _create_examples(self, texts, intents, set_type):
        examples = []
        for i, (text, intent) in enumerate(zip(texts, intents)):
            guid = "{}-{}".format(set_type, i)  
            words = text.split()  
            intent_label = self.intent_labels.index(intent) 
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, ))  # list
        return examples  # list

    def get_examples(self, mode):
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        print("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     set_type=mode)  # list     


processors = {
    "MetaQA-hop-predict": ClsProcessor,
}


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id):
        self.input_ids = input_ids 
        self.attention_mask = attention_mask 
        self.token_type_ids = token_type_ids  
        self.intent_label_id = intent_label_id  

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, 
                                 max_seq_len, 
                                 tokenizer,
                                 pad_token_label_id=-100,  
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type 
    cls_token = tokenizer.cls_token   # [CLS]
    sep_token = tokenizer.sep_token   # [SEP]
    unk_token = tokenizer.unk_token   # [UNK]
    pad_token_id = tokenizer.pad_token_id  # [PAD] -> 0

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = []
        for w in example.words: 
            toks = tokenizer.tokenize(w) 
            tokens.extend(toks)

        # Account for [CLS] and [SEP] 
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        
        # token2id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. 
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length) 
        
        # check length
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        intent_label_id = int(example.intent_label)

        # show some exmples
        if ex_index < 3:
            print("*** Example ***")
            print("guid: %s" % example.guid)
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("intent_label: id = %d" % (intent_label_id))
        
        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          ))
    
    return features  # list


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)  # ClsProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )
    print(cached_features_file)

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")
            
		# add [CLS],[SEP]
        features = convert_examples_to_features(examples, 
                                                args.max_seq_len,
                                                tokenizer,)
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)  

    # Convert to Tensors and build dataset 
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    print('all input ids:',all_input_ids)
    print('all input ids\' size:',all_input_ids.size())

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_intent_label_ids,
    )
    return dataset


def load_tokenizer(args):
    return config.MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)

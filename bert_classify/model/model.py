from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels) 

    def forward(self, x):
        x = self.dropout(x) # x:[batch_size, input_dim]
        return self.linear(x)
    
class ClsBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_list):
        super().__init__(config)  
        self.args = args
        self.num_intent_labels = len(intent_label_list)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)                    
        sequence_output = outputs[0]  
        pooled_output = outputs[1]  
        intent_logits = self.intent_classifier(pooled_output) # size:[batch size, num_intent_labels]
        outputs = ((intent_logits),) + outputs[2:]  # add hidden states and attention if they are here

        # Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))

            outputs = (intent_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    def predict(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)                    
        sequence_output = outputs[0]  
        pooled_output = outputs[1] 
        intent_logits = self.intent_classifier(pooled_output)  # size:[batch size, num_intent_labels]
        outputs = ((intent_logits),) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)

        
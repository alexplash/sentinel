import torch
from transformers import BertTokenizerFast, AutoModel
import torch.nn as nn
import json

class BERT_Arch(nn.Module):

    def __init__(self, bert):

      super(BERT_Arch, self).__init__()

      self.bert = bert

      self.dropout = nn.Dropout(0.1)

      self.relu =  nn.ReLU()

      self.fc1 = nn.Linear(768,512)

      self.fc2 = nn.Linear(512,2)

      self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):

      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict = False)

      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      x = self.fc2(x)

      x = self.softmax(x)

      return x

class PostClassifier:
    def __init__(self, model_path, max_length = 35):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.model = BERT_Arch(bert_model)
        self.model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
        self.model.eval()
        self.max_length = max_length

    def preprocess(self, posts):
        tokens = self.tokenizer.batch_encode_plus(
            posts,
            max_length = self.max_length,
            padding = True,
            truncation = True,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        return tokens
    
    def predict(self, posts):
        tokens = self.preprocess(posts)
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim = 1)
        
        return preds.numpy()
    


    


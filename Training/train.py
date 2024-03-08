import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast

device = torch.device('cuda')

# Change this file path to that of the exported & processed json file from spark.py
file_path = '/Users/alexplash/sentinel/Data/processed_data.json/processed_data.json'
df = pd.read_json(file_path, lines = True)
df.head()

df.shape

df['CLASS'].value_counts(normalize = True)

train_content, temp_content, train_class, temp_class = train_test_split(df['CONTENT'], df['CLASS'],
                                                                   random_state = 2018,
                                                                   test_size = 0.3,
                                                                   stratify = df['CLASS'])

val_content, test_content, val_class, test_class = train_test_split(temp_content, temp_class,
                                                                random_state = 2018,
                                                                test_size = 0.5,
                                                                stratify = temp_class)

# ensure that content is only strings
train_content_lst = train_content.apply(str).tolist()
val_content_lst = val_content.apply(str).tolist()
test_content_lst = test_content.apply(str).tolist()

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

seq_len = [len(i.split()) for i in train_content]
pd.Series(seq_len).hist(bins = 30)

max_seq_len = 35

# tokenize sequences in training set
tokens_train = tokenizer.batch_encode_plus(train_content_lst,
                                           max_length = max_seq_len,
                                           padding = True,
                                           truncation = True,
                                           return_token_type_ids = False
                                           )

# tokenize sequences in validation set
tokens_val = tokenizer.batch_encode_plus(val_content_lst,
                                         max_length = max_seq_len,
                                         padding = True,
                                         truncation = True,
                                         return_token_type_ids = False
                                         )

# tokenize sequence in testing set
tokens_test = tokenizer.batch_encode_plus(test_content_lst,
                                          max_length = max_seq_len,
                                          padding = True,
                                          truncation = True,
                                          return_token_type_ids = False
                                          )

# train set Tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_class.tolist())

# validation set Tensors
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_class.tolist())

# test set Tensors
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_class.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 32

# DataLoader conversion for training set
train_data = TensorDataset(train_seq, train_mask, train_y)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

# DataLoader conversion for validation set
val_data = TensorDataset(val_seq, val_mask, val_y)

val_sampler = SequentialSampler(val_data)

val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)

# freeze all parameters
for param in bert.parameters():
  param.requires_grad = False

# define model architecture

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
    
# pass the pre-trained BERT to our defined architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

from sklearn.utils.class_weight import compute_class_weight

# compute class weights
class_wts = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_class), y = train_class)

print(class_wts)

# convert class weights to tensor
weights = torch.tensor(class_wts, dtype = torch.float)
weights = weights.to(device)

# losee function
cross_entropy = nn.NLLLoss(weight = weights)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr = 1e-3)

# function to train the model
def train():

  model.train()

  total_loss, total_accuracy = 0, 0

  total_preds=[]

  for step,batch in enumerate(train_dataloader):

    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    batch = [r.to(device) for r in batch]

    sent_id, mask, labels = batch

    model.zero_grad()

    preds = model(sent_id, mask)

    loss = cross_entropy(preds, labels)

    total_loss = total_loss + loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    preds=preds.detach().cpu().numpy()

    total_preds.append(preds)

  avg_loss = total_loss / len(train_dataloader)

  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

# define format_time
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}h:{minutes}m:{seconds}s"

# function to evaluate the model
def evaluate():

  print("\nEvaluating...")

  model.eval()

  t0 = time.time()

  total_loss, total_accuracy = 0, 0

  total_preds = []

  for step,batch in enumerate(val_dataloader):

    if step % 50 == 0 and not step == 0:

      elapsed = format_time(time.time() - t0)

      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

    with torch.no_grad():

      preds = model(sent_id, mask)

      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  avg_loss = total_loss / len(val_dataloader)

  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

# train & evaluate the model
epochs = 20

best_valid_loss = float('inf')

train_losses=[]
valid_losses=[]

for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    train_loss, _ = train()

    valid_loss, _ = evaluate()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

# load weights of best model
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))

# get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()

# model's performance
preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))

# confusion matrix
pd.crosstab(test_y, preds)

# change this path to that of your saved_weights.pt, which contains the finalized model values
save_path = '/Users/alexplash/sentinel/Training/saved_weights.pt'
torch.save(model.state_dict(), save_path)
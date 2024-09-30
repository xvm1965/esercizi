
import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import time
import math
import copy
import numpy as np


raw_datasets = load_dataset("glue", "mrpc")     # carica il dataset
checkpoint = "bert-base-uncased"               # carica il modello
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # carica il tokenizer
num_epochs = 3 # imposta il numero di loop di training con lo stesso dataset
batch_size=8 # dimensione del batch
num_batches = math.ceil(raw_datasets["train"].num_rows / batch_size)  # calcola il numero di batches arrotonda per eccesso
num_steps = num_batches * num_epochs # calcola il numero di steps 
metric = load_metric("glue", "mrpc")
warmup_steps =int (num_batches/10) 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

lr_scheduler = get_scheduler(                           # definisce lo scheduler del learning rate
    "linear",                                           # modalità della variazione
    optimizer=optimizer,                                # optimizer
    num_warmup_steps=warmup_steps,                # cicli di warmup
    num_training_steps=num_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #imposta il device, se esiste la GPU device è torch.device (“cuda”) 

progress_bar = tqdm(range(num_steps))

def get_status (unchanging_batches_count, warmup_steps, strategy, epoch, nbatch, loss, min_loss, min_loss_batch, min_loss_epoch, min_loss_strategy, batch_start_time, start_time):
  
  status =f"({unchanging_batches_count:3}/{warmup_steps:3})"
  status += f"{strategy} - epoch: {epoch} batch: {nbatch:3} "
  status += f"loss: {loss.item():.5f} min:{min_loss:.5f}({min_loss_batch:3}/{min_loss_epoch:1}/{min_loss_strategy}) "
  status += f"time: {time.time()- batch_start_time:2.5f}  avg:{(time.time()-start_time)/(nbatch):.5f}"
  return status
  

print ("\n\nFly tokenization training ...")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.to(device) # trasferisce il modello sul device
model.train() #imposta la modalità di funzionamento del modello

start_time=time.time()

# DataLoader con tokenizzazione al volo
on_fly_train_dataloader = DataLoader(
    raw_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=lambda x: {
        **tokenizer(
            [item["sentence1"] for item in x], [item["sentence2"] for item in x], 
            padding=True, truncation=True, return_tensors="pt"
        ),
        'labels': torch.tensor([item["label"] for item in x]) # Pass labels separately
    }
)

best_model_state=None
min_loss = float("inf")
min_loss_batch=None
min_loss_epoch=None
min_loss_strategy = None

strategy = "Fly"
for epoch in range(num_epochs):
    nbatch=1
    unchanging_batches_count=0
    for batch in on_fly_train_dataloader:
        batch_start_time=time.time()

        # Trasferisci il batch sulla GPU (se disponibile)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Training step
        outputs = model(**batch)
        loss = outputs.loss #calcolo della loss
        loss.backward()     #calcolo dei gradienti
        optimizer.step()    #aggiornamento dei pesi
        lr_scheduler.step() #aggiornamento del learning rate
        optimizer.zero_grad() #azzeramento dei gradienti

        if (loss.item()<min_loss):
          min_loss=loss.item()
          min_loss_batch=nbatch
          min_loss_epoch=epoch
          min_loss_strategy = strategy
          best_model_state = copy.deepcopy(model.state_dict())  # Save best model
          unchanging_batches_count = 0
        else:
          unchanging_batches_count +=1
   
        print (get_status (unchanging_batches_count,
                           warmup_steps,
                           strategy, 
                           epoch, 
                           nbatch, 
                           loss, 
                           min_loss, 
                           min_loss_batch, 
                           min_loss_epoch, 
                           min_loss_strategy, 
                           batch_start_time, 
                           start_time))
        
        if unchanging_batches_count >= warmup_steps:
                break  # Exit the inner loop
        nbatch+=1
    # Load best model state
    model.load_state_dict(best_model_state)
    print(f'Epoch {epoch:1} ended - best loss: {min_loss:.5f}({min_loss_batch:3}/{min_loss_epoch:1}/{min_loss_strategy} - {time.time()-batch_start_time:.5f}')
print(f'Fly tokenization ended - best loss: {min_loss:.5f} ({min_loss_batch:3}/{min_loss_epoch:1}/{min_loss_strategy}) - {time.time()-start_time:.5f}')   


print(f'\n\nPre tokenization start')

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.to(device) # trasferisce il modello sul device
model.train() #imposta la modalità di funzionamento del modello


start_time=time.time()
best_model_state=None
min_loss = float("inf")
min_loss_batch=None
min_loss_epoch=None
min_loss_strategy = None

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

pre_tokenize_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

pre_tokenize_train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=pre_tokenize_data_collator
)
pre_tokenize_eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=batch_size, collate_fn=pre_tokenize_data_collator
)

strategy = "Pre"
for epoch in range(num_epochs):
    
    unchanging_batches_count=0
    nbatch=1
    for batch in pre_tokenize_train_dataloader:
        
        batch_start_time=time.time()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if (loss.item()<min_loss):
          min_loss=loss.item()
          min_loss_batch=nbatch
          min_loss_epoch=epoch
          min_loss_strategy=strategy
          best_model_state = copy.deepcopy(model.state_dict())  # Save best model
          unchanging_batches_count = 0
        else:
          unchanging_batches_count +=1

        print (get_status (unchanging_batches_count,
                           warmup_steps,
                           strategy, 
                           epoch, 
                           nbatch, 
                           loss, 
                           min_loss, 
                           min_loss_batch, 
                           min_loss_epoch, 
                           min_loss_strategy, 
                           batch_start_time, 
                           start_time))
        if unchanging_batches_count >= warmup_steps:
                break  # Exit the inner loop
        nbatch+=1
    # Load best model state
    model.load_state_dict(best_model_state)
    print(f'Epoch {epoch:1} ended - best loss: {min_loss:.5f}({min_loss_batch:3}/{min_loss_epoch:1}/{min_loss_strategy} - {time.time()-batch_start_time:.5f}')
print(f'Pre tokenization ended - best loss: {min_loss:.5f} ({min_loss_batch:3}/{min_loss_epoch:1}/{min_loss_strategy}) - {time.time()-start_time:.5f}')   
         
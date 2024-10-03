import torch
from tqdm.auto import tqdm
from transformers import get_scheduler, AdamW, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import time
import math
import copy

# Carica il dataset e il modello
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
batch_size = 8
num_epochs = 3
warmup_steps = int((len (raw_datasets["train"])*0.1) /batch_size)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# optimizer = AdamW(model.parameters(), lr=5e-5)
# lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epochs * len(raw_datasets["train"]))
# initial_model_state = copy.deepcopy(model.state_dict())

# Definisce il device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print (f"\ndevice: {device}\n")

# def get_status(unchanging_batches_count, warmup_steps, moving_avg, strategy, epoch, nbatch, loss, min_loss, min_loss_batch, min_loss_epoch, min_loss_strategy, batch_start_time, start_time):
#     status = f"{unchanging_batches_count:3}/{warmup_steps:3} movavg: {moving_avg:.5f} delta: { 100*abs((moving_avg-loss.item())/moving_avg):5.2f} "
#     status += f"{strategy} - epoch: {epoch} batch: {nbatch:3} "
#     status += f"loss: {loss.item():.5f} min:{min_loss:.5f}({min_loss_batch:3}/{min_loss_epoch:1}/{min_loss_strategy}) "
#     status += f"time: {time.time()-batch_start_time:2.5f} avg:{(time.time()-start_time)/(nbatch):.5f}"
#     return status


def get_status(strategy, epoch, nbatch, loss, moving_avg, batch_start_time, start_time):
    status = f"strategy: {strategy} epoch: {epoch} batch: {nbatch:4} loss: {loss.item():.5f} movavg: {moving_avg:.5f}" 
    status += f"time: {time.time()-batch_start_time:2.5f} avg:{(time.time()-start_time)/(nbatch):2.5f}"
    return status



def train_and_validate(train_dataloader, validation_dataloader, test_dataloader, strategy):
    
    # reset del modello
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epochs * len(train_dataloader))

    start_time = time.time()
    best_model_state = None
    min_loss = float("inf")
    min_loss_batch = None
    min_loss_epoch = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        unchanging_batches_count = 0
        loss_history = []
        nbatch = 1

        for batch in train_dataloader:
            batch_start_time = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}

            # Step di training
            outputs = model(**batch)  #applica il modello al batch
            loss = outputs.loss       #calcola la loss
            loss.backward()           #calcola i gradienti
            optimizer.step()          #aggiorna i pesi ed i parametri
            lr_scheduler.step()       #modifica il passo di aggiornamento
            optimizer.zero_grad()     #azzera i gradienti

            
            # Salva la loss
            loss_history.append(loss.item())
            if len(loss_history) > warmup_steps:
                loss_history.pop(0)
                
            
            moving_avg = sum(loss_history) / min (warmup_steps, nbatch)
            delta = abs((moving_avg-loss.item())/moving_avg)
            if delta < 0.01:
                unchanging_batches_count += 1
            else:
                unchanging_batches_count = 0


            if loss.item() < min_loss:
                min_loss = loss.item()
                min_loss_batch = nbatch
                min_loss_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                
              

            if len (loss_history)==warmup_steps and unchanging_batches_count >= warmup_steps:
                break

            
            print(get_status(strategy, epoch, nbatch, loss, moving_avg, batch_start_time, start_time))
            
            nbatch += 1

        # Validazione
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for val_batch in validation_dataloader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
                total_val_loss += val_loss.item()

                logits = val_outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = val_batch['labels']
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_val_loss = total_val_loss / len(validation_dataloader)
        accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.5f}, Accuracy: {accuracy:.5f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Best validation loss improved to {best_val_loss:.5f}, saving model.")

    # Carica il miglior modello
    model.load_state_dict(best_model_state)
    print(f'{strategy} training ended - best validation loss: {best_val_loss:.5f}')

    #valuta il modello sul dataset di test
    total_val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
            for val_batch in test_dataloader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
                total_val_loss += val_loss.item()

                logits = val_outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = val_batch['labels']
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

    avg_val_loss = total_val_loss / len(test_dataloader)
    accuracy = correct_predictions / total_predictions

    return total_val_loss, correct_predictions, total_predictions, time.time()-start_time

# Prima esecuzione con tokenizzazione al volo
train_dataloader = DataLoader(
    raw_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=lambda x: {
        **tokenizer([item["sentence1"] for item in x], [item["sentence2"] for item in x], padding=True, truncation=True, return_tensors="pt"),
        'labels': torch.tensor([item["label"] for item in x])
    }
)
validation_dataloader = DataLoader(
    raw_datasets["validation"], shuffle=True, batch_size=batch_size, collate_fn=lambda x: {
        **tokenizer([item["sentence1"] for item in x], [item["sentence2"] for item in x], padding=True, truncation=True, return_tensors="pt"),
        'labels': torch.tensor([item["label"] for item in x])
    }
)

test_dataloader = DataLoader(
    raw_datasets["test"], shuffle=True, batch_size=batch_size, collate_fn=lambda x: {
        **tokenizer([item["sentence1"] for item in x], [item["sentence2"] for item in x], padding=True, truncation=True, return_tensors="pt"),
        'labels': torch.tensor([item["label"] for item in x])
    }
)


results = train_and_validate(train_dataloader, validation_dataloader, test_dataloader, "Fly")
# Salva i valori in un dizionario
fly_data_dict = {
    "avg loss": results[0],
    "correct predictions": results[1],
    "total predictions": results[2],
    "time": results[3]
}
# Seconda esecuzione con tokenizzazione precomputata
tokenized_datasets = raw_datasets.map(lambda x: tokenizer(x["sentence1"], x["sentence2"], truncation=True, padding=True), batched=True)

tokenized_datasets = raw_datasets.map(lambda x: tokenizer(x["sentence1"], x["sentence2"], truncation=True, padding=True), batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"]) # Remove inplace argument
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
validation_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))


results = train_and_validate(train_dataloader, validation_dataloader, test_dataloader, "Pre")
# Salva i valori in un dizionario
pre_data_dict = {
    "avg loss": results[0],
    "correct predictions": results[1],
    "total predictions": results[2],
    "time": results[3]
}

print (f"\n{fly_data_dict}\n")
print (f"\n{pre_data_dict}\n")


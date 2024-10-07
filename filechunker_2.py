import logging
# Imposta il livello di logging a ERROR per nascondere gli avvisi
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import importlib.util
# Funzione per verificare se un pacchetto è installato
def install_if_needed(package_name):
    print (f"package: {package_name}")
    if importlib.util.find_spec(package_name) is None:
        print (f"install package: {package_name}")
        !pip install {package_name}
        print (f"package installato: {package_name}")
    else:
        print (f"{package_name} già installato")
# Verifica e installa PyPDF2 se necessario
install_if_needed('PyPDF2')
install_if_needed('transformers')
install_if_needed('torch')
install_if_needed('sentence-transformers')

import os
import spacy
import PyPDF2
import subprocess
import torch
import numpy as np
import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# Inizializza il tokenizer e la pipeline
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
nlp = pipeline("feature-extraction", model=model, tokenizer=tokenizer)


# Funzione per ottenere l'embedding di una frase
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Usa la media dei vettori di embedding come rappresentazione della frase
    embedding = torch.mean(outputs.last_hidden_state, dim=1)
    return embedding.squeeze().numpy()

# Rimuovi la directory esistente se vuoi fare un nuovo clone
repo_dir = "docs"
if os.path.exists(repo_dir):
    !rm -rf {repo_dir}

# Clona il repository
github_token = input ("Token per github: ")
username = "xvm1965"
repo = "docs"
repo_url = f"https://{username}:{github_token}@github.com/{username}/{repo}.git"
!git clone {repo_url}

def preprocess_text(text):
    # 1. Rimuove righe che iniziano con un numero o che sono più corte di 20 caratteri
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()  # Rimuove spazi all'inizio e alla fine della linea
        if len(line) < 20 or re.match(r'^\d+', line):  # Rimuove righe con meno di 20 caratteri o che iniziano con un numero
            continue
        processed_lines.append(line)

    # Ricostruisce il testo dalle righe rimaste
    text = ' '.join(processed_lines)
    
    # 2. Sostituzione di sequenze di caratteri speciali uguali con uno solo (es. "!!!" diventa "!")
    text = re.sub(r'([^\w\s])\1+', r'\1', text)

    # 3. Sostituzione di sequenze di caratteri speciali diversi con uno spazio (es. "!@#" diventa " ")
    text = re.sub(r'[^\w\s]{2,}', ' ', text)

    # 4. Sostituzione di più spazi consecutivi con uno solo
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()  # Rimuove eventuali spazi all'inizio e alla fine del testo

# Funzione per suddividere il testo in frasi basate sui punti finali
def split_text_into_sentences(text):
    sentences = text.split('. ')  # Dividi il testo sulle frasi basate su punto e spazio
    return [s.strip() for s in sentences if s.strip()]  # Rimuovi spazi vuoti e frasi vuote

# Funzione per suddividere il testo in chunk basati su chunk_size e similarità semantica
def split_text_into_chunks(text, chunk_size=512, similarity_threshold=0.8):
    sentences = split_text_into_sentences(text)
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0
    current_embedding = None

   
    for sentence in sentences:
        # Tokenizza la frase e ottieni la lunghezza in token
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)

        # Controllo per suddividere la frase se è più lunga di chunk_size
        if sentence_token_count > chunk_size:
           # Se il chunk corrente ha già delle frasi, lo aggiungiamo a 'chunks' prima di gestire la frase lunga
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_chunk_token_count = 0
            
            # Suddividi la frase in chunk più piccoli
            for i in range(0, sentence_token_count, chunk_size):
                sub_tokens = sentence_tokens[i:i + chunk_size]
                sub_sentence = tokenizer.decode(sub_tokens, skip_special_tokens=True)
                chunks.append(sub_sentence)  # Aggiungi ogni sottosegmento come un chunk separato

            # Vai alla prossima frase, senza fare ulteriori controlli su questa
            continue

        # Ottieni l'embedding per la frase
        sentence_embedding = torch.tensor(nlp(sentence)[0])

       
        

        # Controlla se la frase si adatta nel chunk corrente o se deve iniziarne uno nuovo
        if current_chunk_token_count + sentence_token_count > chunk_size or (
            current_embedding is not None and 
            torch.cosine_similarity(current_embedding.mean(dim=0).unsqueeze(0), sentence_embedding.mean(dim=0).unsqueeze(0)).item() < similarity_threshold
        ):
            # Salva il chunk corrente
            if current_chunk:
                chunks.append(' '.join(current_chunk))

                
            
            current_chunk = [sentence]
            current_chunk_token_count = sentence_token_count
            current_embedding = sentence_embedding  # Imposta l'embedding della nuova frase
        else:
            # Aggiungi la frase al chunk corrente
            current_chunk.append(sentence)
            current_chunk_token_count += sentence_token_count
            
            # Se current_embedding è None (quindi è la prima iterazione), assegniamo direttamente sentence_embedding
            if current_embedding is None:
                current_embedding = sentence_embedding
            else:
                # Altrimenti concateniamo gli embedding
                current_embedding = torch.cat((current_embedding, sentence_embedding), dim=0)

    # Aggiungi l'ultimo chunk se esiste
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# Funzione per estrarre il testo dai PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

# Elenca i file nella directory

pdf_files = [f for f in os.listdir(repo) if f.endswith('.pdf')]

# Itera sui file PDF e stampa i primi 300 caratteri
for pdf_file in pdf_files:
    pdf_path = os.path.join(repo, pdf_file)
    text = extract_text_from_pdf(pdf_path)
    print (f"lunghezza del testo prima del preprocessing {len(text)}")
    text=preprocess_text(text)
    print (f"lunghezza del testo dopo il preprocessing {len(text)}")
    # Suddividi il testo in frasi
    chunks = split_text_into_chunks(text)
    for j, chunk in enumerate(chunks):
        print(f"\n\n{j+1}\n{chunk}")
    

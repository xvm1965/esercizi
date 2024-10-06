import importlib.util
import logging
import os
import spacy
import PyPDF2
import subprocess
import torch
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Imposta il livello di logging a ERROR per nascondere gli avvisi
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Inizializza il tokenizer e la pipeline

# Funzione per verificare se un pacchetto è installato
def install_if_needed(package_name):
    if importlib.util.find_spec(package_name) is None:
        !pip install {package_name}
    
# Verifica e installa PyPDF2 se necessario
install_if_needed('PyPDF2')
install_if_needed('transformers')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

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
github_token = input ("github token: ")
username = "xvm1965"
repo = "docs"
repo_url = f"https://{username}:{github_token}@github.com/{username}/{repo}.git"
!git clone {repo_url}


# Inizializza il tokenizer e la pipeline
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
nlp = pipeline("feature-extraction", model="bert-base-uncased")

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

        # Ottieni l'embedding per la frase
        sentence_embedding = torch.tensor(nlp(sentence)[0])

        # Se la frase è troppo lunga per essere inserita in un chunk, suddividila
        if sentence_token_count > chunk_size:
            # Se il chunk corrente ha già delle frasi, lo aggiungiamo a 'chunks' prima di gestire la frase lunga
            if current_chunk:
                chunks.append(' '.join(current_chunk))  # Salva il chunk corrente
                current_chunk = []  # Reset del chunk corrente

            # Suddividi la frase lunga in pezzi più piccoli di dimensione chunk_size
            for i in range(0, sentence_token_count, chunk_size):
                sub_tokens = sentence_tokens[i:i + chunk_size]  # Prendi una porzione di token
                sub_sentence = tokenizer.decode(sub_tokens, skip_special_tokens=True)  # Converti i token in testo
                chunks.append(sub_sentence)  # Aggiungi ogni sotto-sentence come un chunk separato

        else:
            # Se la frase è semanticamente troppo diversa o il chunk attuale è troppo lungo, salviamo il chunk
            if current_embedding is None or current_chunk_token_count + sentence_token_count > chunk_size or torch.cosine_similarity(current_embedding.mean(dim=0).unsqueeze(0), sentence_embedding.mean(dim=0).unsqueeze(0)).item() < similarity_threshold:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))  # Salva il chunk corrente
                current_chunk = [sentence]  # Inizia un nuovo chunk con la frase corrente
                current_chunk_token_count = sentence_token_count
                current_embedding = sentence_embedding
            else:
                # Aggiungi la frase al chunk corrente
                current_chunk.append(sentence)
                current_chunk_token_count += sentence_token_count
                # Aggiorna l'embedding del chunk corrente
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
    # Suddividi il testo in frasi
    chunks = split_text_into_chunks(text)
    print (f"file: {pdf_file[:50]} - di {len(text):5} caratteri - suddiviso in {len(chunks)} chunk")

import importlib.util
import logging
import os
import spacy
import PyPDF2
import subprocess
from transformers import pipeline
from transformers import AutoTokenizer

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
nlp = pipeline("feature-extraction", model="bert-base-uncased")

# Rimuovi la directory esistente se vuoi fare un nuovo clone
repo_dir = "docs"
if os.path.exists(repo_dir):
    !rm -rf {repo_dir}

# Clona il repository
github_token = input ("Github token: ")

username = "xvm1965"
repo = "docs"
repo_url = f"https://{username}:{github_token}@github.com/{username}/{repo}.git"
!git clone {repo_url}


# Inizializza il tokenizer e la pipeline
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
nlp = pipeline("feature-extraction", model="bert-base-uncased")

def split_text_into_chunks(text, chunk_size=512):
    # Suddivide il testo in frasi
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Usa il tokenizer per contare i token senza passare la frase al modello BERT
        sentence_tokens = tokenizer(sentence, return_tensors='pt', truncation=False)['input_ids'][0]
        sentence_length = len(sentence_tokens)  # Ottieni il numero di token

        # Se l'aggiunta della frase corrente supera il chunk_size, chiudi il chunk attuale
        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        # Se la frase è troppo lunga, la suddividiamo in chunk più piccoli
        if sentence_length > chunk_size:
            split_long_sentence(sentence_tokens, chunk_size, chunks)
        else:
            # Aggiungi la frase al chunk corrente
            current_chunk.append(sentence)
            current_length += sentence_length

    # Aggiungi l'ultimo chunk se non è vuoto
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def split_long_sentence(sentence_tokens, chunk_size, chunks):
    """
    Questa funzione divide una frase lunga in chunk di dimensione massima `chunk_size`
    e li aggiunge alla lista `chunks`.
    """
    # Suddividi i token della frase in chunk più piccoli
    for i in range(0, len(sentence_tokens), chunk_size):
        sub_tokens = sentence_tokens[i:i+chunk_size]  # Prendi i token in chunk
        sub_sentence = tokenizer.decode(sub_tokens, skip_special_tokens=True)  # Converti i token in testo
        chunks.append(sub_sentence)  # Aggiungi ogni sotto-frase al risultato finale


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

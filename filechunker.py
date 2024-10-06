import importlib.util
import os
import spacy
import PyPDF2
import subprocess

# Funzione per verificare se un pacchetto è installato
def install_if_needed(package_name):
    if importlib.util.find_spec(package_name) is None:
        print(f"{package_name} non trovato, installazione in corso...")
        !pip install {package_name}
    else:
        print(f"{package_name} è già installato.")

# Verifica e installa PyPDF2 se necessario
install_if_needed('PyPDF2')
install_if_needed('transformers')


# Carica il modello di linguaggio di spaCy per l'italiano
# Funzione per verificare se un modello spaCy è già installato
def model_installed(model_name):
    try:
        # Controlla se il modello è installato eseguendo un comando spaCy
        subprocess.check_output(f"python -m spacy validate", shell=True)
        return model_name in subprocess.check_output(f"python -m spacy validate", shell=True).decode()
    except subprocess.CalledProcessError:
        return False

# Verifica se il modello 'it_core_news_sm' è già installato, altrimenti installalo
model_name = "it_core_news_sm"
if not model_installed(model_name):
    print(f"Scaricamento del modello '{model_name}'...")
    !python -m spacy download {model_name}
else:
    print(f"Il modello '{model_name}' è già installato.")


nlp = spacy.load('it_core_news_sm')

# Definisci il tuo token GitHub
github_token = "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"
username = "xvm1965"
repo = "docs"  # Nome del repository che contiene i documenti
repo_url = f"https://{username}:{github_token}@github.com/{username}/{repo}.git"

# Verifica se la directory esiste già
if not os.path.exists(repo):
    # Clona il repository su Colab solo se la directory non esiste
    print(f"Clonazione del repository {repo}...")
    !git clone {repo_url}
else:
    # Esegui il pull per aggiornare la copia locale se la directory esiste già
    print(f"La directory '{repo}' esiste già. Eseguo git pull per aggiornare i file...")
    !git -C {repo} pull


# Funzione per suddividere il testo in frasi con spaCy
def split_text_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]  # Suddividi il testo in frasi
    return sentences

# Funzione per estrarre il testo dai PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

# Definisci il percorso alla directory del repository
repo_dir = "docs"  # Il nome del repo clonato

# Elenca i file nella directory
pdf_files = [f for f in os.listdir(repo_dir) if f.endswith('.pdf')]

# Itera sui file PDF e stampa i primi 300 caratteri
for pdf_file in pdf_files:
    pdf_path = os.path.join(repo_dir, pdf_file)
    text = extract_text_from_pdf(pdf_path)
    # Suddividi il testo in frasi
    sentences = split_text_into_sentences(text)
    for j, s in enumerate(sentences):
      print(f"[{j+1:2}] - {s}")
      if j//10==0:break
    print()

import importlib.util
import os

# Funzione per verificare se un pacchetto è installato
def install_if_needed(package_name):
    if importlib.util.find_spec(package_name) is None:
        print(f"{package_name} non trovato, installazione in corso...")
        !pip install {package_name}
    else:
        print(f"{package_name} è già installato.")

# Verifica e installa PyPDF2 se necessario
install_if_needed('PyPDF2')

import PyPDF2

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
    print(f"\nPrimi 300 caratteri di {pdf_file}:\n{text[:300]}")

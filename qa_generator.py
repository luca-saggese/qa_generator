import json
from transformers import pipeline

# Carica i modelli di Hugging Face per la generazione di domande e risposte
qg_pipeline = pipeline("text2text-generation", model="vocabtrimmer/mbart-large-cc25-itquad-qg-trimmed-it")
qa_pipeline = pipeline("question-answering", model="research-backup/mbart-large-cc25-itquad-ae")

# Carica il file JSON con gli articoli del Codice Civile
input_file = "codice-civile.json"  # Modifica con il nome del tuo file
output_file = "domande_risposte_codice_civile.json"

with open(input_file, "r", encoding="utf-8") as f:
    articoli = json.load(f)

# Verifica la struttura dei dati
print(f"📜 Numero totale di articoli: {len(articoli)}")

# Assumi che il JSON abbia una struttura come questa:
# {"articolo": "Art. 2043", "testo": "Risarcimento per fatto illecito: qualsiasi danno ingiusto deve essere risarcito."}

dataset_domande_risposte = []

for articolo in articoli:
    articolo_id = articolo.get("articolo", "N/A")
    testo = articolo.get("testo", "").strip()

    if not testo:
        continue  # Salta gli articoli vuoti

    print(f"📝 Generazione per {articolo_id}...")

    try:
        # Genera 4 domande basate sul testo dell'articolo
        domande = qg_pipeline(testo, max_length=100, num_return_sequences=4)

        for domanda in domande:
            domanda_text = domanda["generated_text"]

            # Usa il modello di QA per estrarre la risposta dal testo
            risposta = qa_pipeline(question=domanda_text, context=testo)

            # Salva nel dataset
            dataset_domande_risposte.append({
                "articolo": articolo_id,
                "domanda": domanda_text,
                "risposta": risposta.get("answer", "N/A")
            })

    except Exception as e:
        print(f"⚠️ Errore con {articolo_id}: {e}")
        continue

# Salva il dataset in un nuovo file JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset_domande_risposte, f, indent=4, ensure_ascii=False)

print(f"✅ Dataset salvato in {output_file}")


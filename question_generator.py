from transformers import pipeline

qg_pipeline = pipeline("text2text-generation", model="vocabtrimmer/mbart-large-cc25-itquad-qg-trimmed-it")

text = "Il dolo implica l'intenzionalità di commettere un reato, mentre la colpa è dovuta a negligenza, imprudenza o imperizia."
question = qg_pipeline(text, max_length=100)

print(question[0]['generated_text'])

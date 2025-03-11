from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="research-backup/mbart-large-cc25-itquad-ae")

context = "Il dolo implica l'intenzionalità di commettere un reato, mentre la colpa è dovuta a negligenza, imprudenza o imperizia."
question = "Qual è la differenza tra dolo e colpa?"

answer = qa_pipeline(question=question, context=context)
print(answer["answer"])

import spacy
from spacy.util import minibatch, compounding
import random
from spacy.training import Example

# Load a blank English model
nlp = spacy.blank("en")

# Add the NER component to the pipeline
ner = nlp.create_pipe("ner")
nlp.add_pipe("ner")

# Add a custom label for the medical terms
ner.add_label("SYMPTOM")

TRAIN_DATA = [
    ("I have been feeling itching and skin rash lately.", {"entities": [(18, 25, "SYMPTOM"), (30, 39, "SYMPTOM")]}),
    # ... more annotated sentences
]

# examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in TRAIN_DATA]
# for batch in minibatch(examples, size=compounding(4.0, 32.0, 1.001)):
#     nlp.update(batch, drop=0.5, losses=losses)


# # Train the model on the annotated data
# optimizer = nlp.begin_training()
# for iteration in range(100):
#     # Shuffle the training data
#     random.shuffle(TRAIN_DATA)
#     losses = {}
    
#     # Batch the data using spaCy's minibatch and compounding utilities
#     batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
#     for batch in batches:
#         texts, annotations = zip(*batch)
#         nlp.update(texts, annotations, drop=0.5, losses=losses)

#     print(f"Iteration {iteration + 1}, Loss: {losses['ner']}")

# # Save the trained model
# # nlp.to_disk("/path/to/your/model")
# # Load the trained model
# # nlp = spacy.load("/path/to/your/model")

# Convert training data to Example objects
examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in TRAIN_DATA]

# Train the model on the Example objects
optimizer = nlp.begin_training()
for iteration in range(100):
    random.shuffle(examples)
    losses = {}
    
    # Batch the data using spaCy's minibatch and compounding utilities
    batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, drop=0.5, losses=losses)

    # print(f"Iteration {iteration + 1}, Loss: {losses['ner']}")

# Predict entities in a new text
text = "I've been experiencing skin rash and some fatigue."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
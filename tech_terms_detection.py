

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers import pipeline
from collections import Counter

nlp_ner = pipeline("ner")

nlp = pipeline("question-answering")
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")  
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")

nlp_BERT = pipeline(
    task="question-answering",
    model=model,
    tokenizer=tokenizer)

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")  
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

nlp_RoBERTa = pipeline(
    task="question-answering",
    model=model,
    tokenizer=tokenizer)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")  
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

nlp_DistillBERT = pipeline(
    task="question-answering",
    model=model,
    tokenizer=tokenizer)

tokenizer = AutoTokenizer.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")  
model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")
nlp_Longformer = pipeline(
    task="question-answering",
    model=model,
    tokenizer=tokenizer)


MAX_N_WORDS_IN_EACH_TECHNICAL_TERM = 4

# Given a transformer model extract terms using it.
def extract_terms(_nlp, sublists_of_words_joined):
  ALL_TECHNICAL_WORDS = []
  for context in sublists_of_words_joined:
    save_this_word = True
    tech_word_with_info = _nlp(question="What is the technical word?", context=context)
    tech_word = tech_word_with_info['answer']
    # print(tech_word_with_info, 'from --->', context)
    score = tech_word_with_info['score']
    if score < 0.15:
      continue

    # If the extracted techword consists of several words take the first n
    subwords_in_tech_word = tech_word.split()

    # If any word in the extracted term belongs to NER words- don't store this term.
    if FILTER_NER_WORDS:
      for word in subwords_in_tech_word:
        if word in NER_WORDS:
          save_this_word = False

    if save_this_word:
      if len(subwords_in_tech_word) <= MAX_N_WORDS_IN_EACH_TECHNICAL_TERM:
        # tech_word = ' '.join(subwords_in_tech_word[:MAX_N_WORDS_IN_EACH_TECHNICAL_TERM])
        ALL_TECHNICAL_WORDS.append(tech_word)
  return ALL_TECHNICAL_WORDS



input_passage = """
Scientists with the XENON1T experiment (shown) observed extra blips in their dark matter detector at low energies. That could be a sign of new particles such as solar axions or from tiny amounts of radioactive tritium, the researchers say. The blips could be explained by weird new particles called solar axions, or unexpected magnetic properties for certain known particles, neutrinos, the researchers propose. Normal particle interactions should have produced around 232 electron recoils at low energy, but the researchers saw 285 — an excess of 53. The XENON1T team suggested that the low-energy events could be due to solar axions, hypothetical particles with no electric charge that could be produced in the sun. This explanation wouldn’t reveal anything new about the universe, but it would be the first time that a detector of this type was sensitive enough to spot such tiny amounts of tritium. Search for new physics with electronic-recoil events in XENON1T. Observation of excess electronic recoil events in XENON1T.
"""

def clean_passage(input_passage):
  input_passage = input_passage.replace("\n", "")

  sublists_of_words_joined = input_passage.split(".") # split at sents (full stops).
  sublists_of_words_joined = [sublist for sublist in sublists_of_words_joined if len(sublist) > 5] # keep sents with at least 5 chars- i.e. remove \n
  return sublists_of_words_joined

# Terms Extracted by RoBERTa
def extract_terms_with_RoBERTa(input_passage):
  sublists_of_words_joined = clean_passage(input_passage)
  return list(map(lambda w: w.upper(), list(set(
    extract_terms(nlp_RoBERTa, sublists_of_words_joined)
    ))))


def extract_terms_with_Ensemble_models(input_passage):
  # Returns list of tuples 

  sublists_of_words_joined = clean_passage(input_passage)

  TERMS = []
  result_RoBERTa = list(set(extract_terms(nlp_Longformer, sublists_of_words_joined)))
  result_BERT = list(set(extract_terms(nlp_DistillBERT, sublists_of_words_joined)))
  result_DistillBERT = list(set(extract_terms(nlp_BERT, sublists_of_words_joined)))
  result_Longformer = list(set(extract_terms(nlp_RoBERTa, sublists_of_words_joined)))

  extracted_terms_with_duplicates = result_RoBERTa  + result_DistillBERT + result_Longformer
  terms_Counter = Counter(extracted_terms_with_duplicates)
  TERMS = terms_Counter.most_common() # list of one tuple ('term', count)
  return list(zip(list(map(lambda t: t[0].upper(), TERMS)), list(map(lambda t: t[1], TERMS))))












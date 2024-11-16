import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math
from cidereval import cider, ciderD

predicted_file_name = "test-predictions.out.txt"
ref_file_name = "test-trgs.given.txt"
chencherry = SmoothingFunction()
with open(predicted_file_name) as f1:
    #predicted_lines = [line.rstrip('\n.') for line in f1]
    predicted_lines = f1.readlines()
with open(ref_file_name) as f2:
    #ref_lines = [line.rstrip('\n.') for line in f2]
    ref_lines = f2.readlines()

result = cider(predictions=predicted_lines, references=ref_lines)

print(round(result["avg_score"]*100,2))
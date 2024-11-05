import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu



def calculate_bleu(epoch,test=False, Warmup=False):
  nltk.download('punkt_tab')
  if test:

    predicted_file_name = "predictions/test-predictions.out.txt"
    ref_file_name = "predictions/test-trgs.given.txt"
    
    
  elif Warmup:
    predicted_file_name = "predictions/warm-predictions.out-"+str(epoch)+".txt"
    ref_file_name = "predictions/warm-trgs.given-"+str(epoch)+".txt"
  
  else:
    predicted_file_name = "predictions/predictions.out-"+str(epoch)+".txt"
    ref_file_name = "predictions/trgs.given-"+str(epoch)+".txt"

  
  with open(predicted_file_name) as f1:
    predicted_lines = [line.rstrip('\n') for line in f1]
  with open(ref_file_name) as f2:
    ref_lines = [line.rstrip('\n') for line in f2]
  
  total = 0
  count = len(predicted_lines)
  for i in range(count):
    hypothesis = word_tokenize(predicted_lines[i])
    reference = word_tokenize(ref_lines[i])
    
    total += sentence_bleu([reference], hypothesis)
  total /= count 
  return round(total*100, 2)
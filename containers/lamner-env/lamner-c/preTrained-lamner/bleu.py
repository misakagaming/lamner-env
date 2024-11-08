import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math



def calculate_bleu(epoch,test=False, Warmup=False, order=4):
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

  
    chencherry = SmoothingFunction()
    with open(predicted_file_name) as f1:
        #predicted_lines = [line.rstrip('\n.') for line in f1]
        predicted_lines = f1.readlines()
    with open(ref_file_name) as f2:
        #ref_lines = [line.rstrip('\n.') for line in f2]
        ref_lines = f2.readlines()


    bleu=[0,0,0]
    curr = 0
    weight=[
        (1./2,1./2),
        (1./3,1./3,1./3),
        (1./4,1./4,1./4,1./4)]
    count = len(predicted_lines)
    count1 = [count, count, count]
    """for i in range(count):
        hypothesis = word_tokenize(predicted_lines[i])
        print(hypothesis)
        reference = word_tokenize(ref_lines[i])
        print(reference)
        bleu = sentence_bleu([reference], hypothesis, smoothing_function=chencherry.method2)
        print(bleu)
        total += bleu"""
    for i in range(count):
        for j in range(3):
            curr = sentence_bleu([word_tokenize(ref_lines[i].strip("\n. "))],
                word_tokenize(predicted_lines[i].strip("\n. ")),
                weights=weight[j],
                smoothing_function=chencherry.method2)
            """if curr < 0.01:
                count1[j] -= 1
                continue"""
            bleu[j] += curr
    return round(bleu[order-1]/count1[order-1]*100, 2)
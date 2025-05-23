import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
import math
import time
from torchtext import data
import torchtext.vocab as vocab
from lamner_utils.utils import set_seed, init_weights, print_log, get_max_lens, count_parameters, calculate_rouge, write_files, epoch_time
from src.attention import Attention
from src.encoder import Encoder
from src.decoder import Decoder
from src.seq2seq import Seq2Seq, train, evaluate, get_preds
from six.moves import map
from src.train_lm import train_language_model
from src.train_ner import train_ner_model
from src.extract_embeds import get_embeds
from run import run_seq2seq
from run_save import run_save

def main():
  set_seed()
  ##Loading parameters for the model
  parser = argparse.ArgumentParser(description="Setting hyperparameters for Lamner")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size to use for seq2seq model")
  parser.add_argument("--embedding_size", type=int, default=512, help="Embedding size to use for seq2seq model")
  parser.add_argument("--hidden_dimension", type=int, default=512, help="Embedding size to use for seq2seq model")
  parser.add_argument("--dropout", type=float, default=0.5, help="Dropout to use for seq2seq model")
  parser.add_argument("--epochs", type=int, default=200, help="Epochs to use for seq2seq model")
  parser.add_argument("--static", type=bool, default=False, help="Keep weigts static after one epoch")
  parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
  parser.add_argument("--duplicates", dest='duplicates', action='store_true', help="Remove Duplicates")
  parser.add_argument("--no-duplicates", dest='duplicates', action='store_false', help="Remove Duplicates")
  parser.add_argument("--code_len", type=int, default=300, help="Set maximum code length")
  parser.add_argument("--comment_len", type=int, default=50, help="Set maximum comment length")
  parser.add_argument('--infer', dest='infer', action='store_true')
  parser.add_argument('--no-infer', dest='infer', action='store_false')
  parser.add_argument('--lam', dest='lam', action='store_true')
  parser.add_argument('--no-lam', dest='lam', action='store_false')
  parser.add_argument('--ner', dest='ner', action='store_true')
  parser.add_argument('--no-ner', dest='ner', action='store_false')
  parser.add_argument('--metric', default="rouge")
  parser.add_argument('--codebert', dest='codebert', action='store_true')
  parser.add_argument('--no-codebert', dest='codebert', action='store_false')
  parser.add_argument("--order", type=int, default=4)
  parser.add_argument("--runsave", type=int, default=0)
  parser.set_defaults(feature=True)
  args = parser.parse_args()
  
  #train_lm-> preprocess code, make correct directories
  print(args.infer)
  #if not(args.infer):
    #train_language_model(args)
    #train_ner_model(args)
    #get_embeds(args)
  if args.runsave == 0:
    run_seq2seq(args)
  else:
    run_save(args)

  
if __name__ == '__main__':
  main()
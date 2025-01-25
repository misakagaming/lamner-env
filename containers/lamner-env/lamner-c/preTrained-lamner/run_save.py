import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
import math
import time
from torchtext import data
import torchtext.vocab as vocab
from lamner_utils.utils import set_seed, init_weights, print_log, get_max_lens, count_parameters, calculate_rouge, write_files, epoch_time, calculate_meteor, calculate_cider
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, ELMoEmbeddings, TransformerDocumentEmbeddings
from src.attention import Attention
from src.encoder import Encoder
from src.decoder import Decoder
from src.seq2seq import Seq2Seq, train, evaluate, get_preds
from six.moves import map
from bleu import calculate_bleu
import numpy as np

def run_save(args):
  set_seed()
  ##Loading parameters for the model
  #parser = argparse.ArgumentParser(description="Setting hyperparameters for Lamner")
  #parser.add_argument("--batch_size", type=int, default=16, help="Batch size to use for seq2seq model")
  #parser.add_argument("--embedding_size", type=int, default=512, help="Embedding size to use for seq2seq model")
  #parser.add_argument("--hidden_dimension", type=int, default=512, help="Embedding size to use for seq2seq model")
  #parser.add_argument("--dropout", type=float, default=0.5, help="Dropout to use for seq2seq model")
  #parser.add_argument("--epochs", type=int, default=200, help="Epochs to use for seq2seq model")
  #parser.add_argument("--static", type=bool, default=False, help="Keep weigts static after one epoch")
  #parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
  #parser.add_argument("--infer", type=bool, default=False, help="Inference")
  #args = parser.parse_args()
  CLIP = 1
  make_weights_static = args.static
  best_valid_loss = float('inf')
  cur_rouge = -float('inf')
  best_rouge = -float('inf')
  test_rouge = 0
  test_bleu = 0
  best_epoch = -1
  MIN_LR = 0.0000001
  MAX_VOCAB_SIZE = 50_000
  early_stop = False
  cur_lr = args.learning_rate
  num_of_epochs_not_improved = 0
  SRC = Field(init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            include_lengths = True)
  TRG = Field(init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
  train_data, valid_data, test_data = data.TabularDataset.splits(
          path='./data_seq2seq/', train='train_seq.csv',
          skip_header=True,
          validation='valid_seq.csv', test='test_seq.csv', format='CSV',
          fields=[('code', SRC), ('summary', TRG)])
  #*****************************************************************************************************
  print("embeds")
  custom_embeddings_semantic_encoder = vocab.Vectors(name = 'custom_embeddings/semantic_embeds.txt',
                                      cache = 'custom_embeddings_semantic_encoder',
                                      unk_init = torch.Tensor.normal_) 
  custom_embeddings_syntax_encoder = vocab.Vectors(name = 'custom_embeddings/syntax_embeds.txt',
                                      cache = 'custom_embeddings_syntax_encoder',
                                     unk_init = torch.Tensor.normal_)
  #custom_embeddings_decoder = vocab.Vectors(name = 'custom_embeddings/decoder_embeddings.txt',
  #                                    cache = 'custom_embeddings_decoder',
  #                                   unk_init = torch.Tensor.normal_)
   #*****************************************************************************************************
  #*****************************************************************************************************
  print("test")
  if args.codebert:
    codebert_embeds = vocab.Vectors(name = 'custom_embeddings/weights.txt',
                      cache = 'codebert_embeds',
                      unk_init = torch.Tensor.normal_) 
  SRC.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = custom_embeddings_semantic_encoder
                   ) 
  TRG.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE 
                     #vectors = custom_embeddings_decoder
                   )			   
  #*****************************************************************************************************
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
          (train_data, valid_data, test_data), 
          batch_size = args.batch_size,
          sort_within_batch = True,
          shuffle=True,
          sort_key = lambda x : len(x.code),
          device = device)
  INPUT_DIM = len(SRC.vocab)
  OUTPUT_DIM = len(TRG.vocab)
  SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
  dim = int(args.embedding_size)
  if not args.lam and not args.ner:
    dim = 0
    if args.codebert:
      dim = 767
  elif not args.lam:
    dim = int(dim - args.embedding_size/2)
    if args.codebert:
      dim += 767
  elif not args.ner:
    dim = int(dim - args.embedding_size/2)
    if args.codebert:
      dim += 767 
  if dim == 0:
    attn = Attention(args.hidden_dimension, args.hidden_dimension)
    enc = Encoder(INPUT_DIM, args.embedding_size, args.hidden_dimension, args.hidden_dimension, args.dropout)
    dec = Decoder(OUTPUT_DIM, args.embedding_size, args.hidden_dimension, args.hidden_dimension, args.dropout, attn)
  else:
    attn = Attention(dim, dim)
    enc = Encoder(INPUT_DIM, dim, dim, dim, args.dropout)
    dec = Decoder(OUTPUT_DIM, dim, dim, dim, args.dropout, attn)
  model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
  model.apply(init_weights)
  #*************************************************************************************
  print_log("Setting Embeddings")
  if args.lam:
    SRC.build_vocab(train_data, 
                   max_size = MAX_VOCAB_SIZE, 
                   vectors = custom_embeddings_semantic_encoder
                 )
    embeddings_enc1 = SRC.vocab.vectors
  if args.ner:
    SRC.build_vocab(train_data, 
	  			 max_size = MAX_VOCAB_SIZE, 
	  			 vectors = custom_embeddings_syntax_encoder
	  		   )
    embeddings_enc2 = SRC.vocab.vectors
  if args.codebert:
    SRC.build_vocab(train_data, 
				 max_size = MAX_VOCAB_SIZE, 
				 vectors = codebert_embeds
			   )
    embeddings_enc3 = SRC.vocab.vectors
  default_embeds = True
  if args.lam:
    if default_embeds:
      default_embeds = False
      embeddings_enc4 = embeddings_enc1
    else:
      embeddings_enc4 = torch.cat([embeddings_enc4, embeddings_enc1], dim=1)
  if args.ner:
    if default_embeds:
      default_embeds = False
      embeddings_enc4 = embeddings_enc2
    else:
      embeddings_enc4 = torch.cat([embeddings_enc4, embeddings_enc2], dim=1)
  if args.codebert:
    if default_embeds:
      default_embeds = False
      embeddings_enc4 = embeddings_enc3
    else:
      embeddings_enc4 = torch.cat([embeddings_enc4, embeddings_enc3], dim=1)
  if not default_embeds:    
    model.encoder.embedding.weight.data.copy_(embeddings_enc4)
  np.savetxt("concat_weigths.txt", embeddings_enc4.cpu().detach().numpy())
#if __name__ == '__main__':
#  main()
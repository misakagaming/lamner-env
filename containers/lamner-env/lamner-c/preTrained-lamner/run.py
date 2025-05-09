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

def run_seq2seq(args):
  set_seed(args.seed)
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
                                      cache = "custom_embeddings_syntax_encoder",
                                     unk_init = torch.Tensor.normal_)
  #custom_embeddings_decoder = vocab.Vectors(name = 'custom_embeddings/decoder_embeddings.txt',
  #                                    cache = 'custom_embeddings_decoder',
  #                                   unk_init = torch.Tensor.normal_)
   #*****************************************************************************************************
  #*****************************************************************************************************
  print("test")
  if args.codebert:
    codebert_embeds = vocab.Vectors(name = 'custom_embeddings/weights.txt',
                      cache = "codebert_embeds",
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
  elif args.codebert:
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
  #embeddings_trg = TRG.vocab.vectors
  #model.decoder.embedding.weight.data.copy_(embeddings_trg)
  #del embeddings_enc1, embeddings_enc3, embeddings_enc2, embeddings_enc3, embeddings_enc4 #embeddings_trg
  #*************************************************************************************
  optimizer = optim.SGD(model.parameters(),lr=args.learning_rate, momentum=0.9)
  TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
  criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
  cd_len = get_max_lens(train_data, test_data, valid_data, code=True)
  sm_len = get_max_lens(train_data, test_data, valid_data, code=False)
  print_log("Maximum Input length is " + str(cd_len) + "... Maximum Output Length is " + str(sm_len))
  print_log("Encoder Vocab Size " + str(INPUT_DIM) + "... Decoder Vocab Size " + str(OUTPUT_DIM))
  print_log("Batch Size:" + str(args.batch_size) + "\nEmbedding Dimension:" + str(args.embedding_size))
  print_log('The model has ' + str(count_parameters(model))+  ' trainable parameters')
  print_log("\nTraining Started.....")
  optimizer.param_groups[0]['lr'] = args.learning_rate
  
  if not(args.infer):
    for epoch in range(args.epochs):
      print("aaaaaaaaaa")
      if MIN_LR>optimizer.param_groups[0]['lr']:
        early_stop = True
        break
  
      if num_of_epochs_not_improved==7:
        #reduce LR
        model.load_state_dict(torch.load('models/best-seq2seq.pt'))
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
        stepLR = optimizer.param_groups[0]['lr']
        num_of_epochs_not_improved = 0
      
      start_time = time.time()
      
      train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
      valid_loss = evaluate(model, valid_iterator, criterion)
      p, t = get_preds(valid_data, SRC, TRG, model, device)
      write_files(p,t,epoch+1)
      if args.metric == "rouge":
        cur_rouge = calculate_rouge(epoch+1)
      elif args.metric == "meteor":
        cur_rouge = calculate_meteor(epoch+1)
      elif args.metric == "cider":
        cur_rouge = calculate_cider(epoch+1)
      elif args.metric == "bleu":
        cur_rouge = calculate_bleu(epoch+1, order=args.order)
      torch.save(model.state_dict(), 'models/seq2seq-'+str(epoch+1)+'.pt')
  
      if best_valid_loss>valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models/best-seq2seq.pt')
        best_epoch = epoch + 1
        num_of_epochs_not_improved = 0
      else:
        num_of_epochs_not_improved = num_of_epochs_not_improved + 1
      
      
      if cur_rouge > best_rouge:
        best_rouge = cur_rouge
      
      if make_weights_static==True:
        model.encoder.embedding.weight.requires_grad=False
        make_weights_static=False
        print_log("Embeddings are static now")
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  
      print_log('Epoch: ' + str(epoch+1) + ' | Time: '+ str(epoch_mins) + 'm' +  str(epoch_secs) + 's')
      print_log('\t Learning Rate: ' + str(optimizer.param_groups[0]['lr']))
      print_log('\t Train Loss: ' + str(round(train_loss, 2)) + ' | Train PPL: ' + str(round(math.exp(train_loss), 2)))
      print_log('\t Val. Loss: ' + str(round(valid_loss, 2 )) + ' |  Val. PPL: '+ str(round(math.exp(valid_loss), 2)))
      if args.metric == "rouge":
        print_log('\t Current Val. Rouge: ' + str(cur_rouge) + ' |  Best Rouge '+ str(best_rouge) + ' |  Best Epoch '+ str(best_epoch))
      if args.metric == "meteor":
        print_log('\t Current Val. METEOR: ' + str(cur_rouge) + ' |  Best METEOR '+ str(best_rouge) + ' |  Best Epoch '+ str(best_epoch))
      if args.metric == "cider":
        print_log('\t Current Val. CIDEr: ' + str(cur_rouge) + ' |  Best CIDEr '+ str(best_rouge) + ' |  Best Epoch '+ str(best_epoch))
      elif args.metric == "bleu":  
        print_log('\t Current Val. BLEU-' + str(args.order) + ': ' + str(cur_rouge) + ' |  Best BLEU-' + str(args.order) + ' '+ str(best_rouge) + ' |  Best Epoch '+ str(best_epoch))
      print_log('\t Number of Epochs of no Improvement '+ str(num_of_epochs_not_improved))

  model.load_state_dict(torch.load('models/best-seq2seq.pt'))
  test_loss = evaluate(model, test_iterator, criterion)
  print_log('Test Loss: ' + str(round(test_loss, 2)) + ' | Test PPL: ' + str(round(math.exp(test_loss), 2)))
  p, t = get_preds(test_data, SRC, TRG, model, device)
  write_files(p,t,epoch=0, test=True)
  if args.metric == "rouge":
    test_rouge = calculate_rouge(epoch=0, test=True)
    print_log('Test Rouge: ' + str(test_rouge))
  if args.metric == "meteor":
    test_rouge = calculate_meteor(epoch=0, test=True)
    print_log('Test METEOR: ' + str(test_rouge))
  if args.metric == "cider":
    test_rouge = calculate_cider(epoch=0, test=True)
    print_log('Test CIDEr: ' + str(test_rouge))
  elif args.metric == "bleu":
      test_bleu = calculate_bleu(epoch=0, test=True, order=args.order)
      print_log('Test BLEU-' + str(args.order) + ': ' + str(test_bleu))
#if __name__ == '__main__':
#  main()
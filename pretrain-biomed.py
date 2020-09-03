#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
import math
import os
import random
import numpy as np
import pickle
import time
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset


from transformers import (
    AutoConfig,
    RobertaForMaskedLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)


# In[2]:


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# In[3]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# In[4]:


def text_to_ids_tensor(file_path,tokenizer):
    start = time.time()
    block_size = tokenizer.max_len-tokenizer.num_special_tokens_to_add(pair=False)
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    text=text[:len(text)//4] # only take 1/4 of the text to train to avoid exceed kernel run time limit
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    examples=[]
    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
        examples.append(
            tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
        )
    print(f"text_to_tensor process time:{int(time.time()-start)}s")
    return torch.tensor(examples)


# In[5]:


def mask_tokens(inputs,tokenizer,args):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    start = time.time()    
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    print(f"mask_token process time: {int(time.time()-start)}s")

    return inputs, labels


# In[6]:


def train(train_dataset, model, tokenizer,args):
    """ Train the model """
    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    print("***** Running training *****")
    print(f"Num examples = {len(train_dataset)}")
    print(f"Num Epochs = {int(args.num_train_epochs)}")
    print(f"Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")

    print(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"Total optimization steps = {t_total}")
    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    
    # Added here for reproductibility
    set_seed(args)

    for epoch in range(int(args.num_train_epochs)):
        print(f"epoch{epoch+1} of {args.num_train_epochs}")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "masked_lm_labels": batch[1]
            }


            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description('(loss=%g)' % loss)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                
                
                # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    print("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    tb_writer.close()
    return global_step, tr_loss / global_step


# In[7]:


def main():
    parser = argparse.ArgumentParser()
    #model arguments
    parser.add_argument("--model_type",default='roberta',type=str)
    parser.add_argument("--model_name_or_path",default='roberta-base',type=str)
    #data arguments
    parser.add_argument("--output_dir",default="./output",type=str)
    parser.add_argument("--train_data_file",default=None,type=str)
    parser.add_argument("--eval_data_file",default=None,type=str)
    parser.add_argument("--mlm_probability",default=0.15,type=float)
    parser.add_argument("--block_size",default=-1,type=int)
    #training arguments
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--max_steps",default=-1,type=int) #If > 0: set total number of training steps to perform. Override num_train_epochs.
    parser.add_argument("--warmup_steps", default=0, type=int) #Linear warmup over warmup_steps.
    parser.add_argument("--logging_steps", type=int, default=-1) #help="Log every X updates steps.
    parser.add_argument("--save_steps", type=int, default=-1) #help="Save checkpoint every X updates steps.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_drop_last", type=bool, default=False)#"Drop the last incomplete batch if it is not divisible by the batch size
    parser.add_argument("--device", type=str, default='cuda')
    args, _= parser.parse_known_args()
    
    logger = logging.getLogger(__name__)
    set_seed(args)
    
    #load config and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # load model weights from vanilla RoBERTa model
    model = RobertaForMaskedLM(
                config=config,
            )
    #Or resume training from saved checkpoint with below code 
    #model.load_state_dict(torch.load("../input/biomedical-questionanswer/roberta-base-pretrain-pubmed8252.bin"))
    model.to(args.device)
    
    # load data and train the model
    print("load the pubmed abstract text and generate dataset")
    file_path='../input/biomedical-questionanswer/abstract.txt'
    inputs_ids=text_to_ids_tensor(file_path,tokenizer)
    inputs_ids,labels=mask_tokens(inputs_ids,tokenizer,args)
    train_dataset = TensorDataset(
                                  inputs_ids,
                                  labels
                                 )
    print("start to train")
    global_step, tr_loss=train(train_dataset, model, tokenizer,args)
    
    #Save the pretrained model
    print("save the model")
    output_dir ="roberta-base-pretrain-pubmed.bin"
    torch.save(model.state_dict(), output_dir)
    
if __name__ == "__main__":
    main()


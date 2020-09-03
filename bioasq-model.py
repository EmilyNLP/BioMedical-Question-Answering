#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import glob
import logging
import os
import random
import timeit
import pickle
import sys
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (
    BertPreTrainedModel,
    RobertaModel,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from data_squad import SquadExample,SquadFeatures,print_feature,SquadResult

from predict_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)


# In[2]:


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# In[3]:


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# In[4]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# In[5]:


def to_list(tensor):
    return tensor.detach().cpu().tolist()


# In[6]:


class RobertaQA(BertPreTrainedModel):
    def __init__(self, config,model_name_or_path=None,pretrained_weights=None):
        super(RobertaQA, self).__init__(config)
        if model_name_or_path:
            self.roberta = RobertaModel.from_pretrained(model_name_or_path, config=config)
        else:
            self.roberta = RobertaModel(config=config)
        if pretrained_weights:
            self.roberta.load_state_dict(pretrained_weights)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        torch.nn.init.normal_(self.qa_outputs.weight, mean=0.0,std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,

        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
  
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits


# In[7]:


def features_to_dataset(features,is_training):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

    if not is_training:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
        )
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_positions,
            all_end_positions,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
        )
    return dataset


# In[8]:


#Since the pretrained roberta model has a LM head, in order to adapt to Q&A model, the weights of LM head layer need to be removed. 
#And adjust the parameter names in the pretrained to the same as the roberta layer in Q&A model. 
def pretrain_parameter_adjust(filepath):    
    state_dict = torch.load(filepath)
    new_keys=[]
    old_keys=[]
    pop_keys=[]
    for key in state_dict.keys():
        if 'roberta.' in key:
            new_key=key.replace('roberta.','')
            new_keys.append(new_key)
            old_keys.append(key)
        else:
            pop_keys.append(key)

    for new_key,old_key in zip(new_keys,old_keys):
        state_dict[new_key]=state_dict.pop(old_key)
    for key in pop_keys:
        state_dict.pop(key)
    return state_dict


# In[9]:


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
    print(f"Instantaneous batch size per GPU ={args.per_gpu_train_batch_size}")

    print(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"Total optimization steps = {int(t_total)}")
    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    
    # Added here for reproductibility
    set_seed(args)

    for epoch in range(int(args.num_train_epochs)):
        print(f"epoch{epoch+1} of {int(args.num_train_epochs)}")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

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
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss / global_step


# In[10]:


def evaluate(model, tokenizer,examples, features,dataset, args,prefix=""):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print(f"Num features = {len(dataset)}")
    print(f"Num examples = {len(examples)}")
    print(f"Batch size = {args.eval_batch_size}")

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]
            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)
    

    evalTime = timeit.default_timer() - start_time
    print(f"Evaluation done in total {int(evalTime)} secs, ({evalTime / len(dataset)} sec per example)")

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


# In[11]:


def main():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type",default='roberta',type=str)
    parser.add_argument("--model_name_or_path",default='roberta-base',type=str)
    parser.add_argument("--pretrained_model_name_or_path",default='../input/biomedical-questionanswer',type=str)

    parser.add_argument("--output_dir",default="./output",type=str)
    parser.add_argument("--train_file",default=None,type=str)
    parser.add_argument("--predict_file",default=None,type=str)
    parser.add_argument("--null_score_diff_threshold",type=float,default=0.0)
    parser.add_argument("--max_seq_length",default=384,type=int)
    parser.add_argument("--doc_stride",default=128,type=int)
    parser.add_argument("--max_query_length",default=64,type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--max_steps",default=-1,type=int) #If > 0: set total number of training steps to perform. Override num_train_epochs.
    parser.add_argument("--warmup_steps", default=0, type=int) #Linear warmup over warmup_steps.
    parser.add_argument("--n_best_size",default=20,type=int) #The total number of n-best predictions to generate in the nbest_predictions.json output file.",)
    parser.add_argument("--max_answer_length",default=30,type=int) #The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.
    parser.add_argument("--logging_steps", type=int, default=500) #help="Log every X updates steps.
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--save_steps", type=int, default=500) #help="Save checkpoint every X updates steps.
    parser.add_argument("--eval_all_checkpoints") #Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose_logging",action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--version_2_with_negative",action="store_true") #help="If true, the SQuAD examples contain some that do not have an answer.",

    args, _= parser.parse_known_args()
    
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    set_seed(args)
    
    #Load the SQuAD dataset from pre-processed features 
    with open("../input/biomedical-questionanswer/features_training.pickle",'rb') as handle:
        train_features=pickle.load(handle)
    train_dataset = features_to_dataset(train_features,is_training=True)
    with open("../input/biomedical-questionanswer/examples_valid.pickle",'rb') as handle:
        valid_examples=pickle.load(handle)
    with open("../input/biomedical-questionanswer/features_valid.pickle",'rb') as handle:
        valid_features=pickle.load(handle)
    valid_dataset=features_to_dataset(valid_features,is_training=False)    
    
    
    #Load the BioASQ dataset from pre-processed features
    with open("../input/biomedical-questionanswer/train_bioasq_features.pickle",'rb') as handle:
        train_bioasq_features=pickle.load(handle)
    train_bioasq_dataset = features_to_dataset(train_bioasq_features,is_training=True)    
    with open("../input/biomedical-questionanswer/test_bioasq_example.pickle",'rb') as handle:
        test_bioasq_example=pickle.load(handle)
    with open("../input/biomedical-questionanswer/test_bioasq_features.pickle",'rb') as handle:
        test_bioasq_features=pickle.load(handle)
    test_bioasq_dataset=features_to_dataset(test_bioasq_features,is_training=False)
    
    #Load config, model, tokenizer
    state_dict=pretrain_parameter_adjust("../input/biomedical-questionanswer/roberta-base-pretrain-pubmed8178.bin")            
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model=RobertaQA(config=config,pretrained_weights=state_dict) #Load from pretrained RoRERTa Model
    #model=RobertaQA(config=config,model_name_or_path=args.model_name_or_path)
    model.to(args.device)
        
    #train on SQuAD dev dataset
    global_step, tr_loss=train(train_dataset, model, tokenizer,args)
    
    #evaluate on SQuAD dev dataset
    results=evaluate(model, tokenizer,valid_examples, valid_features,valid_dataset,args, prefix="SQuAD1")
    print("results on SQuAD dev dataset after fine-tuned with SQuAD train dataset ")
    print(results)
    
    #train the model with BioASQ train dataset
    global_step, tr_loss=train(train_bioasq_dataset, model, tokenizer,args)
    
    #evaluate the model on BioASQ dev dataset
    results_bioasq=evaluate(model, tokenizer,test_bioasq_example, test_bioasq_features,test_bioasq_dataset,args, prefix="BioASQ")
    print("results on BioASQ dev dataset after fine-tuned with SQuAD train dataset and BioASQ train dataset")
    print(results_bioasq)
    
    #evaluate the model(fine-tuned both on SQuAD and BioASQ) on Squad dev dataset
    results_squad_bio_squad=evaluate(model, tokenizer,valid_examples, valid_features,valid_dataset, args,prefix="SQuAD2")
    print("results on SQuAD dev dataset after fine-tuned with SQuAD train dataset and BioASQ train dataset")
    print(results_squad_bio_squad)


# In[12]:


if __name__ == "__main__":
    main()


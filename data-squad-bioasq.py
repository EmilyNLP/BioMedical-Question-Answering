#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import numpy as np
from tqdm import tqdm
import logging
import os
import pickle
from enum import Enum

from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset

MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart"}

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)



def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please select one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the ``truncation`` argument in :meth:`PreTrainedTokenizerBase.__call__`.
    Useful for tab-completion in an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"

class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation. end_position is the real end position of answer(not the list index)
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class SquadFeatures:
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id



def create_examples(input_json, set_type):
    with open(input_json, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    is_training = set_type == "train"
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                is_impossible = qa.get("is_impossible", False) # for squad v2.0
                #print("is_impossible",is_impossible)
                if not is_impossible:
                    if is_training:
                        answer = qa["answers"][0] # only one answer per question in train dataset
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                    else:
                        answers = qa["answers"]  #There are multiple answers per question in Dev dataset
                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)
    return examples

    
def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, tokenizer, is_training):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding="max_length",
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0, # the unique feature id
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"], # the token position in input_ids--->original context position(before tokenzation) 
                start_position=start_position,# the start position in input_ids
                end_position=end_position,# the end position in input_ids
                is_impossible=span_is_impossible,
                qas_id=example.qas_id, # the unque question-answer id from original json file
            )
        )
    return features

def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use

    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    """

    # Defining helper methods
    features = []
    for example in tqdm(examples,total=len(examples),desc="convert example to features", disable=not tqdm_enabled):   
        features_from_one_example=squad_convert_example_to_features(example, 
                                                   max_seq_length, 
                                                   doc_stride, 
                                                   max_query_length, 
                                                   tokenizer, 
                                                   is_training)

        features.append(features_from_one_example)
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    return features
    
def print_feature(feature):
    print('input_ids-----',feature.input_ids)
    print('attention_mask-----',feature.attention_mask)
    print('token_type_ids-----',feature.token_type_ids)
    print('cls_index-----',feature.cls_index)
    print('p_mask-----',feature.p_mask)
    print('example_index-----',feature.example_index)
    print('unique_id-----',feature.unique_id)
    print('paragraph_len-----',feature.paragraph_len)
    print('token_is_max_context-----',feature.token_is_max_context)
    print('tokens-----',feature.tokens)
    print('token_to_orig_map-----',feature.token_to_orig_map)
    print('start_position-----',feature.start_position)
    print('end_position-----',feature.end_position)
    print('is_impossible-----',feature.is_impossible)
    print('qas_id-----',feature.qas_id)

def index_to_str(num):
    return '0'*(3-len(str(num)))+str(num)

def bioasq_to_squad(input_json,training):
    with open(input_json, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["questions"]
    squad_entries=[]
    for entry in tqdm(input_data):
        if entry['type']!='factoid': #only extract the type of 'factoid' question 
            continue
        question=entry['body'].strip()
        if training:
            answer_text=entry['exact_answer'][0].strip()
        else:
            answer_text=entry['exact_answer'][0][0].strip()            
        for index,snippet in enumerate(entry["snippets"]):
            id_num=entry['id']+'_'+index_to_str(index+1)
            context = snippet["text"].strip()
            start=context.find(answer_text)
            if start!=-1:
                answer={"text":answer_text,"answer_start":start}
            else:
                continue
            new_entry={
                       "qas": [
                         {
                          "id": id_num,
                          "question": question,
                          "answers": [answer]
                         }
                         ],
                       "context": context
                       }
            squad_entries.append(new_entry)
    return squad_entries

def entry_to_json(squad_entries,title=None,version=None):
    squad_json={
              "data": [
                {
                  "paragraphs":squad_entries,
                   "title":title
                }],
               "version":version
           }
    return squad_json


class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type",default='roberta',type=str)
    parser.add_argument("--model_name_or_path",default='roberta-base',type=str)
    parser.add_argument("--tokenizer_name",default="",type=str)
    parser.add_argument("--cache_dir",default="",type=str)
    parser.add_argument("--max_seq_length",default=384,type=int)
    parser.add_argument("--doc_stride",default=128,type=int)
    parser.add_argument("--max_query_length",default=64,type=int)
    parser.add_argument("--do_lower_case", action="store_true")

    args, _ = parser.parse_known_args()
    
    tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None)
    
    max_seq_length=args.max_seq_length
    doc_stride=args.max_seq_length
    max_query_length=args.max_query_length
    
    #load SQuAD train dateset, generate train features,  save to future use
    print("Load SQuAD train dateset, generate train features,  save to future use")
    train_examples=create_examples("../input/biomedical-questionanswer/train-v1.1.json", "train")
    is_training=True
    train_features=squad_convert_examples_to_features(
        train_examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        padding_strategy="max_length",
        tqdm_enabled=True)
    with open('features_training.pickle', 'wb') as handle:
        pickle.dump(train_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #load SQuAD dev dateset, generate valid features,  save to future use
    print("Load SQuAD dev dateset, generate valid features,  save to future use")
    valid_examples=create_examples("../input/biomedical-questionanswer/dev-v1.1.json", "valid")
    is_training=False
    valid_features=squad_convert_examples_to_features(
        valid_examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        padding_strategy="max_length",
        tqdm_enabled=True)
    with open('valid_features.pickle', 'wb') as handle:
        pickle.dump(valid_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('valid_examples.pickle', 'wb') as handle:
        pickle.dump(valid_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #convert the BioASQ raw date into the same format as SQuAD, generate features,save for future use
    #train dataset
    print("Convert the BioASQ raw train date into the same format as SQuAD, generate features,save for future use")
    trainining7b_entries=bioasq_to_squad("../input/biomedical-questionanswer/trainining7b.json",training=True)
    trainining7b_squad_json=entry_to_json(trainining7b_entries)
    with open("bioasq7b_squad.json", "w") as outfile:  
        json.dump(trainining7b_squad_json, outfile,indent=2)     
    train_bioasq_example=create_examples("../input/biomedical-questionanswer/bioasq7b_squad.json", "train")
    is_training=True
    train_bioasq_features=squad_convert_examples_to_features(
        train_bioasq_example,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        padding_strategy="max_length",
        tqdm_enabled=True)
    with open('train_bioasq_example.pickle', 'wb') as handle:
        pickle.dump(train_bioasq_example, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('train_bioasq_features.pickle', 'wb') as handle:
        pickle.dump(train_bioasq_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #test dataset
    print("Convert the BioASQ raw test date into the same format as SQuAD, generate features,save for future use")
    test_file_list=["7B1_golden.json","7B2_golden.json","7B3_golden.json","7B4_golden.json","7B5_golden.json"]
    all_test_entries=[]
    i=0
    for file in test_file_list:    
        filepath=os.path.join("../input/biomedical-questionanswer/",file)
        entries=bioasq_to_squad(filepath,training=False)
        all_test_entries+=entries
    all_test_squad_json=entry_to_json(all_test_entries)
    with open("bioasq7b_gold_squad.json", "w") as outfile:  
        json.dump(all_test_squad_json, outfile,indent=2)
    test_bioasq_example=create_examples("../input/biomedical-questionanswer/bioasq7b_gold_squad.json", "test")
    is_training=False    
    test_bioasq_features=squad_convert_examples_to_features(
        test_bioasq_example,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        padding_strategy="max_length",
        tqdm_enabled=True)
    with open('test_bioasq_example.pickle', 'wb') as handle:
        pickle.dump(test_bioasq_example, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('test_bioasq_features.pickle', 'wb') as handle:
        pickle.dump(test_bioasq_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
if __name__=="__main__":
    main()


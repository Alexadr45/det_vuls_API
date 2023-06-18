import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, AutoTokenizer, set_seed
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import pandas as pd
import os
import tree_sitter
from tree_sitter import Language, Parser
import codecs
import torch
import random
import numpy as np
from transformers import RobertaConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import re


base = 'microsoft/unixcoder-base'
model_id = "Model"

tokenizer = AutoTokenizer.from_pretrained(base)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
set_seed(n_gpu)

#Настраиваем парсер для C#
parser = Parser()
CSHARP_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
parser.set_language(CSHARP_LANGUAGE)


#Функция считывания файла 
def file_inner(path): 
    with codecs.open(path, 'r', 'utf-8') as file: 
        code = file.read() 
    return code 
 

def cleaner1(code): 
    """Удаление комментариев в коде, whitespace, приведение к одной строке"""
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)') 
    code = re.sub(pat,'',code) 
    code = re.sub('\r','',code) 
    code = re.sub('\t','',code) 
    code = code.split('\n') 
    code = [line.strip() for line in code if line.strip()] 
    code = ' '.join(code) 
    return(code)


def subnodes_by_type(node, node_type_pattern=''): 
    """Выделение сабнодов с методами в дереве tree-sitter"""
    if re.match(pattern=node_type_pattern, string=node.type, flags=0): 
        return [node] 
    nodes = [] 
    for child in node.children: 
        nodes.extend(subnodes_by_type(child, node_type_pattern = 'method_declaration')) 
    return nodes 


def add_line_delimiter(method): 
    """Разделения кода по строкам"""
    method = method.replace(';', ';\n') 
    method = method.replace('{', '\n{\n') 
    method = method.replace('}', '}\n') 
    return method


def obfuscate(parser, code, node_type_pattern='method_declaration'): 
    """Выделение методов(функций) из кода"""
    tree = parser.parse(bytes(code, 'utf8')) 
    nodes = subnodes_by_type(tree.root_node, node_type_pattern) 
    methods = [] 
    for node in nodes: 
        if node.start_byte >= node.end_byte: 
            continue 
        method = code[node.start_byte:node.end_byte]
        method = add_line_delimiter(method) 
        methods.append(method)
    return methods


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

        
    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
    
    
    def forward(self, attention_mask = None, inputs_embeds = None, output_hidden_states = None, return_dict = True, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions) # attention_mask=input_ids.ne(1)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0] # attention_mask=input_ids.ne(1)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob
            
            
class Input(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 func):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.func=func


class TextData(Dataset):
    def __init__(self, tokenizer, funcs):
        self.examples = []
        for i in tqdm(range(len(funcs))):
            self.examples.append(tokenize_samples(funcs[i], tokenizer))

            
    def __len__(self):
        return len(self.examples)

    
    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), str(self.examples[i].func)


config = RobertaConfig.from_pretrained(base)
config.num_labels = 1
model = RobertaForSequenceClassification.from_pretrained(base, config=config, ignore_mismatched_sizes=True).to(device)
model = Model(model, config, tokenizer)
model.to(device)

#model.load_state_dict(torch.load(pretrain_model_path, map_location=device))
config = PeftConfig.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, model_id)


def cleaner(code):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    code = re.sub('\r','',code)
    code = re.sub('\t','',code)
    code = code.split('\n')
    code = [line + '\n' for line in code if line.strip()]
    code = ''.join(code)
    return(code)


def set_seed(n_gpu, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def tokenize_samples(func, tokenizer, block_size=512):
    clean_func = cleaner(func)
    code_tokens = tokenizer.tokenize(str(clean_func))[:block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return Input(source_tokens, source_ids, clean_func)


def clean_special_token_values(all_values, padding=False):
    # special token in the beginning of the seq
    all_values[0] = 0
    if padding:
        # get the last non-zero value which represents the att score for </s> token
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        # special token in the end of the seq
        all_values[-1] = 0
    return all_values


def get_word_att_scores(all_tokens: list, att_scores: list) -> list:
    word_att_scores = []
    for i in range(len(all_tokens)):
        token, att_score = all_tokens[i], att_scores[i]
        word_att_scores.append([token, att_score.cpu().detach().numpy()])
    return word_att_scores


def get_all_lines_score(word_att_scores: list):
    # verified_flaw_lines = [''.join(l) for l in verified_flaw_lines]
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]
    # to return
    all_lines_score = []
    score_sum = 0
    line_idx = 0
    flaw_line_indices = []
    line = ""
    for i in range(len(word_att_scores)):
        # summerize if meet line separator or the last token
        if ((word_att_scores[i][0] in separator) or (i == (len(word_att_scores) - 1))) and score_sum != 0:
            score_sum += word_att_scores[i][1]
            all_lines_score.append(score_sum)
            line = ""
            score_sum = 0
            line_idx += 1
        # else accumulate score
        elif word_att_scores[i][0] not in separator:
            line += word_att_scores[i][0]
            score_sum += word_att_scores[i][1]
    return all_lines_score


def find_vul_lines(tokenizer, inputs_ids, attentions):
    ids = inputs_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    original_lines = ''.join(all_tokens).split("Ċ")

    # take from tuple then take out mini-batch attention values
    attentions = attentions[0][0]
    attention = None
    # go into the layer
    for i in range(len(attentions)):
        layer_attention = attentions[i]
        # summerize the values of each token dot other tokens
        layer_attention = sum(layer_attention)
        if attention is None:
            attention = layer_attention
        else:
            attention += layer_attention
    # clean att score for <s> and </s>
    attention = clean_special_token_values(attention, padding=True)
    # attention should be 1D tensor with seq length representing each token's attention value
    word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)
    all_lines_score = get_all_lines_score(word_att_scores)
    all_lines_score_with_label = sorted(range(len(all_lines_score)), key=lambda i: all_lines_score[i], reverse=True)
    return all_lines_score_with_label


def predict(model, tokenizer, funcs, device, best_threshold=0.5, do_linelevel_preds=True):

    check_dataset = TextData(tokenizer, funcs)
    check_sampler = SequentialSampler(check_dataset)
    check_dataloader = DataLoader(check_dataset, sampler=check_sampler, batch_size=1, num_workers=0)

    model.to(device)
    model.eval()
    methods = {}
    for idx, batch in enumerate(check_dataloader, start=1):
        method = []
        inputs_ids =  batch[0].to(device)
        func = batch[1]
        with torch.no_grad():
            # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
            logit, attentions = model(input_ids=inputs_ids, output_attentions=True)
            pred = logit.cpu().numpy()[0][1] > best_threshold
            if pred:
                vul_lines = find_vul_lines(tokenizer, inputs_ids, attentions)
                method.append({'orig_func': func})
                method.append({'predict': 1})
                method.append({'vul_lines': vul_lines[:10]})
            else:
                vul_lines = None
                # method.append({'orig_func': func})
                # method.append({'predict': 0})
            methods[('method ' + str(idx))] = method

    return methods

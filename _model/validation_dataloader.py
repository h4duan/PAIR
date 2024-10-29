from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import torch
import numpy as np
import wandb
import ast
from transformers import LogitsProcessor, LogitsProcessorList, AutoTokenizer
import compress_json
import re
from utils import *
import obonet
from sklearn.metrics import f1_score
import torch.distributed as dist

def filter_ec_incomplete(my_list):
    my_list = ast.literal_eval(my_list)
    return [s for s in my_list if (s.count('.') >= 3 and "n" not in s)]

def flatten(matrix):
    return [item for row in matrix for item in row]

def contains_english_characters(input_string):
    for char in input_string:
        if char.isalpha():
            return True
    return False

def filter_batch(prefix, sequences, labels):
    index = []
    for ii, label in enumerate(labels):
        if prefix in label:
            index += [ii]
    filter_s = [sequences[ii] for ii in index]
    filter_l = [labels[ii] for ii in index]
    return filter_s, filter_l

def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    new_column = torch.full((shifted_input_ids.size(0), 1), decoder_start_token_id).to(input_ids.device)
    shifted_input_ids = torch.cat((new_column, shifted_input_ids), dim=1)
    return shifted_input_ids

class RestrictiveLogitsProcessor(LogitsProcessor):
    def __init__(self, token_trie, prefix, tokenizer, separator=None):
        self.token_trie = token_trie
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.separator = separator

    def __call__(self, input_ids, scores):
        # Set scores of all tokens that are not in the allowed list to -infinity
        for ii, prediction in enumerate(input_ids):
            #print(prediction)
            #print(self.tokenizer.batch_decode(prediction, skip_special_tokens=True))
            if len(prediction) > len(self.prefix) + 1:
                p = prediction[len(self.prefix)+1:].tolist()
                if p[-1] == 13 or p[-1] == self.tokenizer.eos_token_id:
                    #disallowed_token_ids = set(range(scores.shape[1])) - set(allowed_token_ids)
                    scores[ii] = float('-inf')
                    scores[ii][self.tokenizer.eos_token_id] = 0
                    continue
                #print(p)
                #print(self.tokenizer.batch_decode(p, skip_special_tokens=True))
                p = [str(pp) for pp in p]
            #print(p)
                p = " ".join(p)
            else:
                p = " "
            
            #pri            
            #print(self.tokenizer.batch_decode(p.split, skip_special_tokens=True))
            try:
                allowed_token_ids = [int(ss) for ss in self.token_trie[p]]
            except:
                print(p)
                allowed_token_ids = [13]
                #print(self.token_trie[" ".join(p.split()[:-1])])
                
                #exit()
            disallowed_token_ids = set(range(scores.shape[1])) - set(allowed_token_ids)
            scores[ii, list(disallowed_token_ids)] = float('-inf')
            #print(torch.sort(scores[ii]))
            #print("-----")
        return scores


def last_occurrence(lst, element):
        # Reverse the list and find the index of the last occurrence
    last_index = len(lst) - 1 - lst[::-1].index(element)
    return last_index


class RestrictiveLogitsProcessorEC(LogitsProcessor):
    def __init__(self, token_trie, prefix, tokenizer, separator=None):
        self.token_trie = token_trie
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.separator = 11

    def __call__(self, input_ids, scores):
        # Set scores of all tokens that are not in the allowed list to -infinity
        for ii, prediction in enumerate(input_ids):
            #print(prediction)
            #print(self.tokenizer.batch_decode(prediction, skip_special_tokens=True))
            if len(prediction) > len(self.prefix) + 1:
                p = prediction[len(self.prefix)+1:].tolist()
                if self.separator not in p:
                    p = [str(pp) for pp in p] 
                    p = " ".join(p)
                elif p[-1] == self.separator:
                    p = " "
                else:
                    last_index = last_occurrence(p, self.separator)
                    p = p[last_index+1:]
                    p = [str(pp) for pp in p]
                    p = " ".join(p)
                    #print(p)
            else:
                p = " "
            
            try:
                allowed_token_ids = [int(ss) for ss in self.token_trie[p]]
            except:
                print(p)
                #print(prediction)
                continue
                #return scores
                #exit()

            disallowed_token_ids = set(range(scores.shape[1])) - set(allowed_token_ids)
            scores[ii, list(disallowed_token_ids)] = float('-inf')
            #print(torch.sort(scores[ii]))
            #print("-----")
        return scores

class RestrictiveLogitsProcessorGo(LogitsProcessor):
    def __init__(self, token_trie, prefix, tokenizer, end_separator=None, eos=None):
        self.token_trie = token_trie
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.end_separator = tokenizer(end_separator).input_ids[0]
        self.eos = tokenizer(eos).input_ids[0]
        #print(self.end_separator, self.eos)
        #exit()

    def __call__(self, input_ids, scores):
        # Set scores of all tokens that are not in the allowed list to -infinity
        for ii, prediction in enumerate(input_ids):
            #print(prediction)
            #print(self.tokenizer.batch_decode(prediction, skip_special_tokens=True))
            if len(prediction) > len(self.prefix) + 1:
                p = prediction[len(self.prefix)+1:].tolist()
                if p[-1] == self.end_separator:
                    p = " "
                elif p[-2:] == self.eos:
                    #disallowed_token_ids = set(range(scores.shape[1])) - set(allowed_token_ids)
                    scores[ii] = float('-inf')
                    scores[ii][self.tokenizer.eos_token_id] = 0
                    continue
                elif self.end_separator in p:
                    last_index = last_occurrence(p, self.end_separator)
                    p = p[last_index:]
                    p = [str(pp) for pp in p]
                    p = " ".join(p)
                else:
                    p = [str(pp) for pp in p]
            #print(p)
                    p = " ".join(p)
            else:
                p = " "
            
            allowed_token_ids = [int(ss) for ss in self.token_trie[p]]

            disallowed_token_ids = set(range(scores.shape[1])) - set(allowed_token_ids)
            scores[ii, list(disallowed_token_ids)] = float('-inf')
            #print(torch.sort(scores[ii]))
            #print("-----")
        return scores



class RemoveLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        # Set scores of all tokens that are not in the allowed list to -infinity
        
        #print(scores)
        #exit()
        disallowed_token_ids = set(range(scores.shape[1])) - set(self.allowed_token_ids)
        scores[:, list(disallowed_token_ids)] = float('-inf')
        return scores

"""
def load_name_tokens(path_to_frames, decoder_name, name_preprocessor):
    ec_frame = compress_json.load(path_to_frames+"protein_recommended_name_frame.json.gz")
    all_ec_token_ids = set()
    batch_ec_number = set()
    fids = list(ec_frame.keys())
    batch_size = 500
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    all_ec_numbers = set()
    for ii, fid in enumerate(fids):
        key = "recommended_name"
        name = name_preprocessor.preprocess_name(ec_frame[fid]["content"][key])
        if len(name) == 0:
            continue
        name = " " + name + "."
        batch_ec_number.add(name)
        all_ec_numbers.add(name)
        if ii > 0 and (ii % batch_size == 0 or ii == len(fids) - 1):
            batch_ec_number = list(batch_ec_number)
            tokens = tokenizer([batch_ec_number[0]]).input_ids[0]
            all_ec_token_ids.update(flatten(tokenizer(batch_ec_number).input_ids))
            batch_ec_number = set()
    return list(all_ec_numbers), all_ec_token_ids
"""

def load_name_tokens(path_to_frames, decoder_name, name_preprocessor):
    ec_frame = compress_json.load(path_to_frames+"protein_recommended_name_frame.json.gz")
    all_ec_token_ids = set()
    batch_ec_number = set()
    fids = list(ec_frame.keys())
    batch_size = 500
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    all_ec_numbers = set()
    ec_trie = {}
    for ii, fid in enumerate(fids):
        key = "recommended_name"
        name = name_preprocessor.preprocess_name(ec_frame[fid]["content"][key])
        #if "bert" in decoder_name:
        #    name = name_preprocessor.preprocess_name(ec_frame[fid]["content"][key]) + "."
            #print(name)
        if len(name) == 0:
            continue
        #if name.startswith("flavodoxin"): 
        #    print(name)
            #exit()
        name = " " + name
        batch_ec_number.add(name)
        all_ec_numbers.add(name)
        if ii > 0 and (ii % batch_size == 0 or ii == len(fids) - 1):
            batch_ec_number = list(batch_ec_number)
            batch_ec_token = tokenizer(batch_ec_number).input_ids
            for ec_token in batch_ec_token:
                ec_token = [str(ec) for ec in ec_token]
                #print(ec_token)
                for token_p in range(len(ec_token)):
                    if token_p == 0:
                        ec_num = " "
                    else:
                        ec_num = " ".join(ec_token[:token_p])
                    if ec_num in ec_trie:
                        #print(ec_num, int(ec_token[token_p+1]))
                        ec_trie[ec_num].add(int(ec_token[token_p]))
                    else:
                        ec_trie[ec_num] = set([int(ec_token[token_p])])
                ec_token = " ".join(ec_token)
                if ec_token in ec_trie:
                    ec_trie[ec_token].add(13)
                else:
                    ec_trie[ec_token] = set([13])
            #all_ec_token_ids.update(flatten(tokenizer(batch_ec_number).input_ids))
            batch_ec_number = set()
    #print(ec_trie)
    #print(len(ec_trie))
    #exit()
    return list(all_ec_numbers), ec_trie


def flatten(l):
    return [item for sublist in l for item in sublist]

def load_ec_tokens(path_to_frames, decoder_name):
    ec_frame = compress_json.load(path_to_frames+"protein_ec_frame.json.gz")
    all_ec_token_ids = set()
    batch_ec_number = set()
    fids = list(ec_frame.keys())
    batch_size = 500
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    all_ec_numbers = set()
    all_ec_tokens = set()
    ec_trie = {}
    for ii, fid in enumerate(fids):
        key = "ec_numbers"
        for ec in ec_frame[fid]["content"][key]:
            if "n" in ec:
                continue
            ec_n = ec.split(".")
            if len(ec_n) < 4:
                continue
            ec_n = " ".join(ec_n)
            ec_1 = " " + ec_n + ","
            batch_ec_number.add(ec_1)
            all_ec_numbers.add(ec_1)
            ec_2 = " " + ec_n + "."
            batch_ec_number.add(ec_2)
            all_ec_numbers.add(ec_2)
        if ii > 0 and (ii % batch_size == 0 or ii == len(fids) - 1):
            batch_ec_number = list(batch_ec_number)
            batch_ec_token = tokenizer(batch_ec_number).input_ids
            #print(batch_ec_token)
            for ec_token in batch_ec_token:
                all_ec_tokens.update(ec_token)
                ec_token = [str(ec) for ec in ec_token]
                #print(ec_token)
                for token_p in range(len(ec_token)):
                    if token_p == 0:
                        ec_num = " "
                    else:
                        ec_num = " ".join(ec_token[:token_p])
                    if ec_num in ec_trie:
                        #print(ec_num, int(ec_token[token_p+1]))
                        ec_trie[ec_num].add(int(ec_token[token_p]))
                    else:
                        ec_trie[ec_num] = set([int(ec_token[token_p])])
            #all_ec_token_ids.update(flatten(tokenizer(batch_ec_number).input_ids))
            batch_ec_number = set()
    #print(all_ec_tokens)
    #print(len(all_ec_tokens))
    #exit()
    #exit()
    return list(all_ec_tokens), ec_trie


def load_go_tokens(go_graph, decoder_name, subontology):
    #go_graph = obonet.read_obo(path_to_go)
    all_go = ["."]
    all_go_tokens = set()
    batch_size = 500
    batch_go_token = set(["."])
    go_trie = dict()
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    for ii, node in enumerate(go_graph):
        #print(node)
        if node[1]["namespace"] != subontology:
            #print(node[1]["namespace"])
            continue
        go = " \"" + node[1]["name"] + "\""
        all_go += [go, go+"."]
        batch_go_token.add(go)
        batch_go_token.add(go+".")
        if ii > 0 and (ii % batch_size == 0 or ii == len(go_graph) - 1):
            batch_go_token = tokenizer(list(batch_go_token)).input_ids
            for go_token in batch_go_token:
                go_token = [str(go) for go in go_token]
                #print(ec_token)
                for token_p in range(len(go_token)):
                    if token_p == 0:
                        go_num = " "
                    else:
                        go_num = " ".join(go_token[:token_p])
                    if go_num in go_trie:
                        #print(ec_num, int(ec_token[token_p+1]))
                        go_trie[go_num].add(int(go_token[token_p]))
                    else:
                        go_trie[go_num] = set([int(go_token[token_p])]) 
            #all_go_tokens.update(flatten(tokenizer(batch_go_token).input_ids))
            batch_go_token = set()
    #print(all_go)
    #exit()
    print(f"{subontology} has {len(go_trie)} entries")
    return list(all_go), go_trie



def filter_go_label(go_label):
    if go_label[:3] == "GO:":
        return go_label[3:]


def preprocess_name_list(my_list):
    #new_names = ast.literal_eval(my_list)
    #print(my_list, len(my_list))
    #new_names = [preprocess_name(s) for s in my_list]
    #new_names = [s for s in new_names if len(s) > 0]
    return preprocess_name(my_list)

def preprocess_seq_list(my_list):
    my_list = ast.literal_eval(my_list)
    #print(my_list, len(my_list
    return my_list[0]

class TestDataset(Dataset):
    """Protein test dataset for binary property prediction."""

    def __init__(self, dataset, category):
        if category == "names":
            category = "recommended_name_raw"
        self.data = dataset[["pid", "seq", category]]
        #print(len(self.data))
        self.data = self.data.dropna(subset=category)
        #print(len(self.data))
        self.data["seq"] = self.data["seq"].apply(preprocess_seq_list)
        self.data = self.data[self.data["seq"].apply(len) <= 1024]
        self.category = category
        if category == "EC":
            self.data[category] = self.data[category].apply(filter_ec_incomplete)
            self.data = self.data[self.data[category].apply(len)>=1]
        if category == "recommended_name_raw":
            self.name_preprocessor = name_preprocessor("/work1/maddison/haonand/biochem_frames/")
            self.category = "recommended_name_raw"
            self.data  = self.data[self.data[category].notna()]
            self.data[category] = self.data[category].apply(self.name_preprocessor.preprocess_name)
            self.data = self.data[self.data[category].apply(len)>0]


    def __len__(self):
        """__len__."""
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        #print(row)
        if self.category == "EC":
            return row["pid"], row["seq"], row[self.category]
        if self.category == "recommended_name_raw":
            #print(row)
            return row["pid"], row["seq"], str(row[self.category])
        if self.category == "GO":
            return row["pid"], row["seq"], [filter_go_label(l) for l in ast.literal_eval(row[self.category])]
        if self.category == "loc":
            return row["pid"], row["seq"], row["loc"]


def postprocess_ec(all_ec_numbers, prefix, decode_outputs, tokenizer, num_beam, top_k=10):
    outputs = decode_outputs.sequences
    scores = decode_outputs.sequences_scores
    predicted_ec = []
    for ii in range(len(scores) // num_beam):
        output = outputs[ii*num_beam:(ii+1)*num_beam]
        score = scores[ii*num_beam:(ii+1)*num_beam].detach().cpu().numpy()
        sorted_indices = np.argsort(score)
        topk_indices = sorted_indices[::-1]
        if len(topk_indices) > top_k:
            topk_indices = topk_indices[:top_k]
        topk_output = [output[jj] for jj in topk_indices]
        topk_prediction = tokenizer.batch_decode(output, skip_special_tokens=True)
        best_prediction = ""
        for jj in range(top_k):
            try:
                prediction = " ".join(topk_prediction[jj][len(prefix)+1:].split()[:4])
            except:
                print("invalid format")
                continue
            if prediction in all_ec_numbers:
                best_prediction = prediction
                break
        predicted_ec += [best_prediction]
    return predicted_ec


def postprocess_ec(all_ec_numbers, prefix, decode_outputs, tokenizer, num_beam, top_k=10):
    outputs = decode_outputs.sequences
    scores = decode_outputs.sequences_scores
    predicted_ec = []
    top_k = num_beam
    for ii in range(len(scores) // num_beam):
        output = outputs[ii*num_beam:(ii+1)*num_beam]
        score = scores[ii*num_beam:(ii+1)*num_beam].detach().cpu().numpy()
        sorted_indices = np.argsort(score)
        topk_indices = sorted_indices[::-1]
        if len(topk_indices) > top_k:
            topk_indices = topk_indices[:top_k]
        topk_output = [output[jj] for jj in topk_indices]
        topk_prediction = tokenizer.batch_decode(output, skip_special_tokens=True)
        best_prediction = ""
        for jj in range(top_k):
            try:
                prediction = " ".join(topk_prediction[jj][len(prefix)+1:].split()[:4])
            except:
                print("invalid format")
                continue
            #print(prediction)
            #print(all_ec_numbers[:10], len(all_ec_numbers))
            if prediction in all_ec_numbers:
                best_prediction = prediction
                break
        predicted_ec += [best_prediction]
    return predicted_ec

def find_prefix(prediction, go_terms):
    label = ""
    label_index = 0
    for ii, go in enumerate(go_terms):
        if prediction.startswith(go) and len(go) > len(label):
            label = go
            label_index = ii
    return label, label_index


#import numpy as np

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def postprocess_go(all_go_anno, prefix, decode_outputs, tokenizer, num_beam, name2term):
    outputs = decode_outputs.sequences
    scores = decode_outputs.sequences_scores
    #softmax = torch.nn.Softmax()
    scores = torch.sigmoid(decode_outputs.sequences_scores * 0.5)
    num_protein = len(scores) // num_beam
    all_predictions = []
    all_scores = np.zeros((num_protein, len(all_go_anno)))
    top_k = num_beam
    for ii in range(num_protein):
        output = outputs[ii*num_beam:(ii+1)*num_beam]
        score = scores[ii*num_beam:(ii+1)*num_beam].detach().cpu().numpy()
        sorted_indices = np.argsort(score)
        topk_indices = sorted_indices[::-1]
        if len(topk_indices) > top_k:
            topk_indices = topk_indices[:top_k]
        topk_output = [output[jj] for jj in topk_indices]
        topk_prediction = tokenizer.batch_decode(output, skip_special_tokens=True)
        #print(topk_prediction)
        #exit()
        go_predictions = {}
        for jj in range(top_k):
            try:
                prediction = topk_prediction[jj][len(prefix)+1:]
            except:
                print("invalid format")
                continue
            #modify_prediction, prediction_index = find_prefix(prediction, all_go_anno)
            if len(prediction) > 0:
                if prediction.endswith("."):
                    prediction = prediction[:-1]
                if prediction not in name2term:
                    print(prediction)
                    print("prediction not in go ontology!")
                    continue
                go_term = name2term[prediction]
                if go_term not in go_predictions:
                    go_predictions[go_term] = score[jj]
                else:
                    go_predictions[go_term] = max(score[jj], go_predictions[go_term])
                #go_predictions += [prediction]
                #if score[jj] > all_scores[ii][prediction_index]:
                    #print(score[jj])
                #    all_scores[ii][prediction_index] = score[jj]
        all_predictions.append(go_predictions)
    return all_predictions

def broadcast_and_concatenate_tensor(tensor, src=0, dim=0):
    # Broadcast the tensor from the source process to all other processes
    #tensor = dist.broadcast(tensor, src=src)

    # Gather tensors from all processes onto the source process
    gathered_tensors = [torch.zeros_like(tensor).reshape(1) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor)

    # Concatenate the tensors along the specified dimension
    #print(gathered_tensors)
    concatenated_tensor = torch.cat(gathered_tensors, dim=dim)

    return concatenated_tensor


def compute_f1(prediction, label):
    intersection_set = label & prediction
    if len(intersection_set) == 0:
        return 0
    precision = len(intersection_set) / len(prediction)
    recall = len(intersection_set) / len(label)
    return (2 * precision * recall) / (precision + recall)


def eval_ec(model, dataloader, cfg, all_ec_numbers, ec_trie, mode="val", rank=0):
    #print(rank)
    acc = 0
    total = 0
    prefix = "This protein has the EC number:"
    prefix_token = model.decoder_tokenizer([prefix], return_tensors="pt", padding=False,
                                             max_length=model.decoder_max_len, truncation=True)

    model.eval()
    #logits_processor = LogitsProcessorList([
    
    #RemoveLogitsProcessor(allowed_token_ids=all_ec_numbers)
    #])
    logits_processor = LogitsProcessorList([
        RestrictiveLogitsProcessorEC(token_trie=ec_trie, prefix=prefix_token[0], tokenizer=model.decoder_tokenizer)
    ])
    #file1 = open("ec_predition_1205_3.txt", "w")
    #print(len(dataloader.dataset))
    all_pids = []
    all_labels = []
    for ii, data in enumerate(dataloader):
        if mode == "val":
            pids, sequences, labels = data
            sequences = list(sequences)
        else:
            sequences, labels = data["x"], data["y"]
            sequences = flatten(sequences)
        labels = list(labels)
        if mode == "train":
            sequences, labels = filter_batch(prefix, sequences, labels)
            if len(sequences) == 0:
                continue
        #all_pids += pids
        #all_labels += labels
        #continue
        """
        if 'A8ZXW5' in pids:
            print("here!!")
            
            exit()
        else:
            continue
        """
        #print(len(sequences))
        #exit()
        total += len(sequences)
        #print(total)
        protein_tokens = model.encoder_tokenizer(sequences, return_tensors="pt", padding=True, max_length = model.encoder_max_len, truncation=True)
        protein_inputids = protein_tokens.input_ids.to("cuda")
        protein_attentionmask = protein_tokens.attention_mask.to("cuda")
        prefix_b = [prefix] * len(labels)
        prompts_tokens = model.decoder_tokenizer(prefix_b, return_tensors="pt", padding=False, max_length = model.decoder_max_len, truncation=True)
        prompts_inputids = prompts_tokens.input_ids.to("cuda")
        prompts_inputids = shift_tokens_right(prompts_inputids, model.model.module.config.decoder_start_token_id)
        
        
        #if cfg.train.eval_only:
        #    with torch.no_grad():
        #        outputs = model.model.module.generate(eos_token_id=13, output_scores=True, do_sample=cfg.decode.sample, input_ids = protein_inputids, min_new_tokens = 4, logits_processor=logits_processor, renormalize_logits=True, decoder_input_ids = prompts_inputids, attention_mask = protein_attentionmask,  max_new_tokens=cfg.decode.ec_max_length, num_beams=cfg.decode.ec_num_beams, temperature=cfg.decode.ec_temperature)
        #else:
        with torch.no_grad():
            outputs = model.model.module.generate(eos_token_id=13, renormalize_logits = True, do_sample=False, input_ids = protein_inputids, min_new_tokens = 4, decoder_input_ids = prompts_inputids, attention_mask = protein_attentionmask,  max_new_tokens=cfg.decode.ec_max_length, num_beams=cfg.decode.ec_num_beams)

        outputs = model.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
   
        for jj in range(len(labels)):
            label = set(labels[jj])
           
            prediction = outputs[jj][len(prefix)+1:-1]
            #prediction_2 = outputs_2[jj][len(prefix)+1:-1]
            #print(outputs[jj])
            #print(outputs_2[jj])
            if mode == "train":
                label = label[len(prefix)+1:]
            try:
                if mode == "val":
                    if "," in prediction:
                        #print("comma")
                        #print(prediction)
                        prediction = prediction.split(", ")
                        #print(prediction)
                    else:
                        prediction = [prediction]
                    prediction = set([".".join(pp.split()) for pp in prediction])


            except:
                prediction = set()
            
            #print(pids[jj], prediction, label, file=file1)
            if "n" in label:
                print("bad n !!!")
                print(label)
                exit() 
            #print(sequences[jj])
            
            #print("no n")
            print(prediction, compute_f1(prediction, label))
            #print(prediction_2, compute_f1(prediction_2, label))
            print(label)
            print("-----")
            #exit()
            
            
            #if prediction == label:
            f1 = compute_f1(prediction, label)
            #print(f1, total)
            #print("-----")
            acc += f1
            
        
        
        print(acc / total, total)
        if total >= cfg.test.ec_test_size:
            break
    #df = pd.DataFrame({'pid': all_pids, 'EC': all_labels})
    #df.to_csv("ec_val.csv", sep="\t", index=False)
    #print(all_pids)
    #print(all_labels)
    #exit()
    if cfg.train.parallel == "ddp":
        acc = torch.tensor(acc / total).to("cuda")
            #print(acc)
        acc = broadcast_and_concatenate_tensor(acc)
            #print(acc)
        score = torch.mean(acc).item()
        if rank == 0 and mode == "train":
            wandb.log({"EC train f1": score})
        elif rank == 0:
            wandb.log({"EC val f1": score})
        return score
    #exit()
    else:
        acc = acc / total
        if mode == "train":
            wandb.log({"EC train f1": acc})
        else:
            wandb.log({"EC val f1": acc})
        print(f"EC: {acc}")
    #if mode == "train":
    
    #wandb.log({"EC train accuracy": acc})
    #else:
    #    wandb.log({"EC val accuracy": acc})
        return acc

def substring_list(name, lists):
    for l in lists:
        if name in l or l in name:
            return True
    return False


import re

def remove_space_around_punctuation(input_string):
    # Define a regex pattern to match spaces before and after punctuation
    pattern = r'\s*([-/.,;:?!])\s*'

    # Use re.sub() to replace matches with the punctuation without spaces
    result = re.sub(pattern, r'\1', input_string)
    if input_string != result:
        print(input_string)
        print(result)
        print("---------")
    return result


def eval_name(model, dataloader, cfg, name_trie=None, mode="val", rank=0):
    acc_exact = 0
    acc_substring = 0
    total = 0
    prefix = "The name of the protein is"
    model.eval()
    predictions = []
    targets = []
    prefix_token = model.decoder_tokenizer([prefix], return_tensors="pt", padding=False,
                                             max_length=model.decoder_max_len, truncation=True)

    model.eval()
    
    logits_processor = LogitsProcessorList([
        RestrictiveLogitsProcessor(token_trie=name_trie, prefix=prefix_token[0], tokenizer=model.decoder_tokenizer)
    ])

    
    acc = 0
    #print(name_trie)
    #print("205" in name_trie, "103" in name_trie)
    #exit()
    #logits_processor = LogitsProcessorList([
    #    RemoveLogitsProcessor(allowed_token_ids=name_trie)
    #])

    for ii, data in enumerate(dataloader):
        if mode == "val":
            pids, sequences, labels = data
            sequences = list(sequences)
        else:
            sequences, labels = data["x"], data["y"]
            sequences = flatten(sequences)
        labels = list(labels)
        if mode == "train":
            sequences, labels = filter_batch(prefix, sequences, labels)
            if len(sequences) == 0:
                continue
        total += len(sequences)
        print(total)
        protein_tokens = model.encoder_tokenizer(sequences, return_tensors="pt", padding=True,
                                                 max_length=model.encoder_max_len, truncation=True)
        protein_inputids = protein_tokens.input_ids.to("cuda")
        protein_attentionmask = protein_tokens.attention_mask.to("cuda")
        prefix_b = [prefix] * len(labels)
        prompts_tokens = model.decoder_tokenizer(prefix_b, return_tensors="pt", padding=True,
                                                 max_length=model.decoder_max_len, truncation=True)
        #print(model.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True))
        prompts_inputids = prompts_tokens.input_ids.to("cuda")
        if "bert" in cfg.train.decoder_name:
            prompts_inputids = prompts_inputids[:, :-1]
        #print(model.decoder_tokenizer.batch_decode(prompts_inputids, skip_special_tokens=False))
        #exit()
        prompts_inputids = shift_tokens_right(prompts_inputids, model.model.module.config.decoder_start_token_id)
        if "gpt" in cfg.train.decoder_name:
            with torch.no_grad():
                outputs = model.model.module.generate(eos_token_id=13, min_new_tokens = 1, do_sample=cfg.decode.name_sample, input_ids=protein_inputids,
                                                  renormalize_logits=True, decoder_input_ids=prompts_inputids, 
                                                  attention_mask=protein_attentionmask, early_stopping=False,
                                                  max_new_tokens=cfg.decode.name_max_length,
                                                  num_beams=cfg.decode.name_num_beams, temperature=cfg.decode.name_temperature)
        else:
            with torch.no_grad():
                outputs = model.model.module.generate(length_penalty=1.0,min_new_tokens = 1, do_sample=cfg.decode.name_sample, input_ids=protein_inputids,
                                                  renormalize_logits=True, decoder_input_ids=prompts_inputids, 
                                                  attention_mask=protein_attentionmask, early_stopping="never",
                                                  max_new_tokens=cfg.decode.name_max_length,
                                                  num_beams=cfg.decode.name_num_beams, temperature=cfg.decode.name_temperature)

        #print(outputs)
        #exit()
        #predicted_go, score_go = postprocess_go(all_go_anno, prefix, outputs, model.decoder_tokenizer, cfg.decode.go_num_beams)

        #outputs = postprocess_go
        outputs = model.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print(model.decoder_tokenizer.pad_token_id, model.decoder_tokenizer.eos_token_id)
        #print(outputs)
        #exit()
        for jj in range(len(outputs)):
            label = labels[jj]
            if mode == "train":
                label = preprocess_name(label[len(prefix)+1:])
            if len(label) == 0:
                continue
            try:
                prediction = outputs[jj][len(prefix)+1:-1]
                if "bert" in cfg.train.decoder_name:
                    prediction = remove_space_around_punctuation(prediction)
            except:
                prediction = " "
            
            predictions += [prediction]
            targets += [label]
            
            """
            print(prediction)
            
            print(label)
            print("------")
            """

            #prediction = prediction[
            #print(targets) 
            if prediction == label:
                acc += 1
        if rank == 0:
            print(acc / total, total)
            
        #print(compute_name_similarity(predictions, targets))
        #exit()
        """
        if total >= cfg.test.name_test_size:
            exact, substring, bleu = compute_name_similarity(predictions, targets)
            #exit()
            wandb.log({f"name {mode} exact match": exact})
            wandb.log({f"name {mode} substring match": substring})
            wandb.log({f"name {mode} bleu": bleu})
            return substring
        """
    exact, substring, bleu = compute_name_similarity(predictions, targets)
    if cfg.train.parallel == "ddp":
        torch.distributed.barrier()
        exact = torch.tensor(exact).to("cuda")
        exact = broadcast_and_concatenate_tensor(exact)
        exact = torch.mean(exact).item()

        substring = torch.tensor(substring).to("cuda")
        substring = broadcast_and_concatenate_tensor(substring)
        substring = torch.mean(substring).item()

        bleu = torch.tensor(bleu).to("cuda")
        bleu = broadcast_and_concatenate_tensor(bleu)
        bleu = torch.mean(bleu).item()
    if rank == 0:
        wandb.log({f"name {mode} exact match": exact})
        wandb.log({f"name {mode} substring match": substring})
        wandb.log({f"name {mode} bleu": bleu})
    return exact


"""
def compute_f1(prediction, label):
    intersection_set = label & prediction
    if len(intersection_set) == 0:
        return 0
    precision = len(intersection_set) / len(prediction)
    recall = len(intersection_set) / len(label)
    return (2 * precision * recall) / (precision + recall)
"""
def compute_fmax(prediction_score, label_score):
    max_score = 0
    #print(np.shape(prediction_score), np.shape(label_score))
    for threshold in np.linspace(0.200, 0.700, num=100):
        hard_prediction = np.zeros_like(prediction_score)
        mask = prediction_score > threshold
        hard_prediction[mask] = 1
        score = f1_score(label_score, hard_prediction, average="samples")
        #print(score, threshold, np.min(prediction_score), np.max(prediction_score))
        if score > max_score:
            max_score = score
    return max_score


def eval_go(model, dataloader, cfg, all_go_anno, go_trie, mode="val", subontology = "", go_graph=None, name2term=None):
    acc = 0
    total = 0
    old_total = 0
    acc_p = 0
    prefix = "This protein has the GO annotations:"
    model.eval()
    #file1 = open("go_prediction.txt", "w")
    prefix_token = model.decoder_tokenizer([prefix], return_tensors="pt", padding=True,
                                                 max_length=model.decoder_max_len, truncation=True)
    logits_processor = LogitsProcessorList([
        RestrictiveLogitsProcessorGo(token_trie=go_trie, prefix=prefix_token[0], tokenizer=model.decoder_tokenizer, end_separator="\"", eos = "\".")
    ])
    prediction_score = []
    label_score = []
    go_network = obonet.read_obo("http://release.geneontology.org/2023-01-01/ontology/go-basic.obo")
    for ii, data in enumerate(dataloader):
        if mode == "val":
            pids, sequences, labels = data
            sequences = list(sequences)
        else:
            sequences, labels = data["x"], data["y"]
            sequences = flatten(sequences)
        labels = list(labels)
        #total += len(sequences)
        #print(sequences)
        protein_tokens = model.encoder_tokenizer(sequences, return_tensors="pt", padding=True,
                                                 max_length=model.encoder_max_len, truncation=True)
        protein_inputids = protein_tokens.input_ids.to(model.device_list[0])
        protein_attentionmask = protein_tokens.attention_mask.to(model.device_list[0])
        prefix_b = [prefix] * len(labels)
        prompts_tokens = model.decoder_tokenizer(prefix_b, return_tensors="pt", padding=True,
                                                 max_length=model.decoder_max_len, truncation=True)
        prompts_inputids = prompts_tokens.input_ids.to(model.device_list[0])
        prompts_inputids = shift_tokens_right(prompts_inputids, model.model.module.config.decoder_start_token_id)
        with torch.no_grad():
            outputs = model.model.module.generate(eos_token_id=1911, decoder_input_ids = prompts_inputids, do_sample=False, input_ids=protein_inputids, renormalize_logits=True,  
                                                  attention_mask=protein_attentionmask, early_stopping=True,
                                                  max_new_tokens=cfg.decode.go_max_length, length_penalty=1.0,
                                                  num_beams=cfg.decode.go_num_beams, temperature=cfg.decode.go_temperature)
        #predicted_go, score_go = postprocess_go(all_go_anno, prefix, outputs, model.decoder_tokenizer, cfg.decode.go_num_beams)
        #prediction_score += score_go
        #print(outputs)
        #exit()
        predicted_go = model.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print(predicted_go)
        #exit()
        #predicted_go = postprocess_go(all_go_anno, prefix, outputs, model.decoder_tokenizer, cfg.decode.go_num_beams, name2term)
        #outputs = [12]
        #outputs = model.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        """
        for jj, prediction in enumerate(predicted_go):
            for go, score in prediction.items():
                print(pids[jj], go, score, file=file1)
        """
        
        for jj in range(len(labels)):
            label = labels[jj]
            #print(predicted_go[jj])
            prediction = predicted_go[jj][len("This protein has the GO annotations: "):-1].split("\" \"")
            if mode == "train":
                label = label[len("This protein has the GO annotations: "):].split(", ")
            else:
                go_label = []
                for anno in label:
                    go_term = f"GO:{anno}"
                    if go_term not in go_graph or go_graph[go_term]["namespace"] != subontology:
                        continue
                    #try:
                    go_label += [go_term]
                    
            #except:
            #            go_label += [""]
                label = go_label
            if len(label) == 0:
                continue
            prediction[0] = prediction[0][1:]
            prediction[-1] = prediction[-1][:-1]
            #prediction = [s[1:-1] for s in prediction]
            prediction_go = set(prediction)
            prediction = []
            for gg in prediction_go:
                if gg in name2term:
                    prediction += [name2term[gg]]
            #print(len(prediction))
            prediction = get_all_go_ancestors(go_network, prediction)
            #print(len(prediction))
            label = set(label)
            #if ii ==  0:
            print(prediction)
            print(label)
            #print(compute_f1(prediction, label))
            #print("-----")
            acc += compute_f1(prediction, label)
            total += 1
            print(acc/total, total)
        if total >= cfg.test.go_test_size:
            acc = acc / total
            #fmax = compute_fmax(np.asarray(prediction_score), np.asarray(label_score))
            wandb.log({f"go {subontology} f1 score": acc})
            #wandb.log({f"go {mode} fmax score": fmax})
            return acc
    

def eval_deeploc(model, dataloader):
    acc = 0
    model.eval()
    location = ['Cell membrane', 'Cytoplasm', 'Endoplasmic reticulum',
       'Golgi apparatus', 'Lysosome/Vacuole', 'Mitochondrion', 'Nucleus',
       'Peroxisome', 'Plastid', 'Extracellular']
    prompts = [l.lower() for l in location]
    for ii, data in enumerate(dataloader):
        sequences, loc = data
        sequences = list(sequences)
        scores = [0] * len(prompts)
        for jj, p in enumerate(prompts):
            with torch.no_grad():
                input_tokens = model.encoder_tokenizer(sequences, return_tensors="pt", padding=True,
                                                      max_length=model.encoder_max_len, truncation=True,
                                                      return_attention_mask=True)
                input_ids = input_tokens.input_ids.to(model.device_list[0])
                input_attentionmask = input_tokens.attention_mask.to(model.device_list[0])
                output_tokens = model.decoder_tokenizer([p] * len(sequences), return_tensors="pt", padding=True,
                                                       max_length=model.decoder_max_len, truncation=True)
                output_ids = output_tokens.input_ids.to(model.device_list[0])
                #print(len(output_ids[0]), p)
                output_attentionmask = output_tokens.attention_mask.to(model.device_list[0])
                scores[jj] = torch.mean(model.model(input_ids=input_ids, labels=output_ids, attention_mask=input_attentionmask, decoder_attention_mask=output_attentionmask).loss).item() * len(input_ids[0])
        print(scores)
        max_index = np.argmin(scores)
        print(max_index)
        #if location[max_index] == loc[0]:
        print("correct")
        print(location[max_index], loc)
        if location[max_index] == loc[0]:
            acc += 1
        if ii > 200:
            break
    return acc / 200




import sys
from tqdm import tqdm
from datetime import datetime
import math
import ast
import wget
import numpy as np
import yaml
import random
import compress_json
from datetime import datetime
from operator import itemgetter
import gzip
import xmltodict
import re
from collections import deque, Counter
import json
from _fact.protein_ec.parser import *
import os
import torch
import networkx
import obonet
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import nltk
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, Sampler
#from catalyst.data.dataset.torch import DatasetFromSampler
from typing import Iterator, List, Optional, Union
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)



class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))



def get_file_path_and_date_from_key( key, paths ):
    date = None
    for fin in paths:
        if key in fin:
            path = paths[fin]
            date = fin[-7:]
    if date is None: raise Exception( "File is not listed in paths:", key  )
    return path, date

def parse_sprot_facts( fact_type_name, file_path, file_date, source, fact_fn, pid_table ):

    data = {}
    I = 0
    with gzip.open(file_path, "rt") as fin:
          entry = ""
          for line in fin:
              entry = line if line.startswith("<entry") else entry + line
              if line.startswith("</entry"):
                  protein = xmltodict.parse(entry, xml_attribs=True)
                  I += 1
                  #get pid and deprecated pids
                  pids = protein["entry"]["accession"]
                  if type(pids) == list:
                      pid = pids[0] #other pids are deprecated
                      deprecated_pids = pid[1:]
                  else:
                      pid = pids
                      deprecated_pids = None

                  #get latest date to be the fact date
                  fact_date = get_date( [file_date, pid_table[pid]["date"] ]  )

                  #create subject list with this entry's protein
                  subjects = [ "pid@" + pid ]

                  facts_attr = fact_fn( protein  )
                  # the above returns a list since a uniprot entry can generate multiple facts
                  for attr in facts_attr:
                      if "subjects" in attr:
                          subjects = attr["subjects"]
                      if "content" in attr:
                          data[ random.getrandbits(128) ] = { "fact_type": fact_type_name , "subjects": subjects, "content": attr["content"],
                                                     "date": fact_date, "source": source
                                                     }

                  if I % 1000 == 0: print(I)
                  #if len(data) == 100:
                  #    return data
    return data

def ec_num_to_text(paths):
    table = { }
    deg123_path,_ = get_file_path_and_date_from_key( "enzyme-ec-deg123"  , paths  )
    with open(deg123_path) as f:
        data = f.readlines()
    for line in data:
        if line[0].isdigit():
            info = line[:-1].split('  ')
            ec_num, name = info[0], info[-1]
            ec_num = ec_num.replace(" ", "").replace("-", "").split(".")
            ec_num = [e for e in ec_num if e != ""]
            name = name.strip()
            if len(ec_num) == 1:
                ec_num = ec_num[0]
                ec_1 = "This enzyme is a type of {}.".format(name[:-2].lower())
                table[ec_num] = ec_1
            elif len(ec_num) == 2:
                ec_2 = table[ec_num[0]][:-1] + ', ' + name[0].lower() + name[1:]
                ec_num = '.'.join(ec_num)
                table[ec_num] = ec_2
            elif len(ec_num) == 3:
                ec_2 = table['.'.join(ec_num[:2])]
                ec_3 = ec_2 + " " + name
                ec_num = '.'.join(ec_num)
                table[ec_num] = ec_3
    deg4_path,_ = get_file_path_and_date_from_key( "enzyme-ec-deg4"  , paths  )
    with open(deg4_path) as f:
         data = f.readlines()
    ec_num = None
    text = None
    for line in data:
        if len(line) > 2:
            if line[:3] == "ID ":
                ec_num = re.findall(r'\d+\.\d+\.\d+\.\d+', line)
                if ec_num:
                    ec_num = ec_num[0]
            if line[:3] == "DE " and ec_num:
                text = ' '.join(line.split()[1:])
                ec_3 = '.'.join(ec_num.split('.')[:-1])
                table[ec_num] = table[ec_3] + " This enzyme is a "  + text
    return table

def print_nested_dict(d, indent=0):
    for key, value in d.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent + 4)
        else:
            print(" " * (indent + 4) + str(value))

def anonymize_prompt( prompt, names ):
    prompt = prompt.lower()
    for name in names:
        prompt = prompt.replace(name, "this protein")
    return prompt

def generate_organism_name( organism  ):
    organism = list(organism.values())
    random.shuffle(organism)
    return organism[0].lower()

def random_flip_aaseq(seq):
    if(random.randint(0,1)): return seq
    return seq[::-1]

def generate_aaseq( table_entry, augment_with_variants, augment_with_isoforms, return_all=False):
    seqs = [ table_entry["aaseq"]  ]
    if augment_with_variants and table_entry["variants"]:
        #print("variants")
        for var in table_entry["variants"]:
            seqs.append( var["aaseq"]  )
    if augment_with_isoforms and table_entry["isoforms"]:
        #print("isoforms")
        for iso in table_entry["isoforms"]:
            if augment_with_isoforms:
                seqs.append( table_entry["isoforms"][iso]["aaseq"]  )
            if augment_with_variants and augment_with_isoforms and table_entry["isoforms"][iso]["variants"]:
                for var in table_entry["isoforms"][iso]["variants"]:
                    seqs.append( var["aaseq"]  )
    #print(len(seqs))
    if return_all:
        return seqs
    random.shuffle(seqs)
    return seqs[0]

def get_date(dates):
    timestamps = [ datetime( int(date.split('-')[0]), int(date.split('-')[1]), 1  ).timestamp() for date in dates  ]
    i, _ = max(enumerate(timestamps), key=itemgetter(1))
    return dates[i]

def get_test_set( pid_table, paths ):
    test_set = {}
    train_pids = set(pid_table.keys())
    file_path,_ = get_file_path_and_date_from_key("sprot", paths)
    with gzip.open(file_path, "rt") as fin:
           entry = ""
           for line in fin:
               entry = line if line.startswith("<entry") else entry + line
               if line.startswith("</entry"):
                   protein = xmltodict.parse(entry, xml_attribs=True)
                   if type(protein["entry"]["accession"]) == str: protein["entry"]["accession"] = [protein["entry"]["accession"]]
                   assert type(protein["entry"]["accession"]) == list
                   pid = protein["entry"]["accession"][0]
                   pids = set(protein["entry"]["accession"])
                   if len(train_pids & pids) == 0:
                       test_set[pid] = { "ec": get_protein_ec_from_uniprot( paths, protein  )[0]  }
    return test_set

def process_uniref50(paths):
    uniref_path,_ = get_file_path_and_date_from_key( "uniref50", paths  )
    pid2cluster = {}
    cluster2pid = {}
    xml = ""
    counter = 0
    with gzip.open(uniref_path, "rt") as fileobject:
        for line in tqdm(fileobject):
            if line.startswith("<entry"):
                xml = line
            else:
                xml += line
                if line.startswith("</entry"):
                    counter += 1
                    uniref_cluster = xmltodict.parse(xml, xml_attribs=True)
                    properties = uniref_cluster['entry']['property']
                    seen_pids = set()
                    for p in properties:
                        if p['@type'] == 'member count':
                            cluster_size = int(p['@value'])
                    cluster_center = uniref_cluster['entry']['representativeMember']
                    found_id = False
                    if cluster_center['dbReference']['@type'] != 'UniProtKB ID': # some are ids from uniparc; checked a few, other members of cluster also only have uniparc ids
                        continue
                    for p in cluster_center['dbReference']['property']:
                        if p['@type'] == "UniProtKB accession" and not found_id: # uniref stores active and dep pids; seems like first one is always active
                            found_id = True
                            cluster_center_pid = p['@value']
                            assert cluster_center_pid not in cluster2pid, "cluster centre is already in dict, check this case: {}".format(cluster_center_pid)
                            assert cluster_center_pid not in pid2cluster, "cluster centre is already in dict, check this case: {}".format(cluster_center_pid)

                            cluster2pid[cluster_center_pid] = [cluster_center_pid]
                            pid2cluster[cluster_center_pid] = cluster_center_pid
                            seen_pids.add(cluster_center_pid)
                    if not found_id: assert cluster_size == 1, uniref_cluster # if there is no uniprot id, assert nothing else exists in cluster

                    if cluster_size > 1:
                        other_cluster_members = uniref_cluster['entry']['member']
                        if type(other_cluster_members) == dict: # there is only one other member, wrap in list
                            other_cluster_members = [other_cluster_members]
                        for protein in other_cluster_members:
                            if protein['dbReference']['@type'] != 'UniProtKB ID':
                                cluster_size -= 1 # if there is no uniprot id, decrease cluster size for assert statement
                                continue
                            found_id = False
                            for p in protein['dbReference']['property']:
                                if p['@type'] == "UniProtKB accession" and not found_id: # uniref stores active and dep pids; seems like first one is always active one
                                    pid = p['@value']
                                    found_id = True
                                    assert pid not in pid2cluster, "PID is already in dict, check this case: {}".format(pid)
                                    assert pid not in seen_pids, "PID is already in seen pids"
                                    pid2cluster[pid] = cluster_center_pid
                                    cluster2pid[cluster_center_pid].append(pid)
                                    seen_pids.add(pid)
                            assert found_id, protein
                    assert len(seen_pids) == cluster_size, "The number of pids processed for cluster != member count!!! {} {}".format(len(seen_pids), cluster_size)
    return pid2cluster, cluster2pid

def expand_test_seqs(test_pids, cluster2pid, pid2cluster):
    print("there are {} test pids".format(len(test_pids)))
    pids_to_remove = set()
    counter = 0
    for pid in test_pids:
        pids_to_remove.add(pid)
        if pid in pid2cluster:
            cluster_pid = pid2cluster[pid]
            #pids_to_remove.add(pid)
            pids_to_remove.update(cluster2pid[cluster_pid])
        else:
            counter +=1
    print("there are {} pids total in test clusters".format(len(pids_to_remove)))
    print("num pids not found in UniRef50", counter)
    return pids_to_remove

def remove_pubmed_substrings(s):
    # The regular expression looks for ( followed by pubmed: and then any sequence of characters that is not ) until it finds )
    s = re.sub(r'\(pubmed:[^)]+\)', '', s, flags=re.I)
    return s

def remove_strain_substrings(s):
    # The regular expression looks for ( followed by pubmed: and then any sequence of characters that is not ) until it finds )
    s = re.sub(r'\(strain[^)]+\)', '', s, flags=re.I)
    return s

def remove_similarity_substrings(s):
    # The regular expression looks for ( followed by pubmed: and then any sequence of characters that is not ) until it finds )
    s = re.sub(r'\(By similarity\)', '', s, flags=re.I)
    return s

def remove_step_substrings(s):
    # The regular expression looks for ( followed by pubmed: and then any sequence of characters that is not ) until it finds )
    pattern = re.compile(r': step \d+/\d+.')
    result_string = re.sub(pattern, '', s)
    #s = re.sub(r'\(By similarity\)', '', s, flags=re.I)
    return result_string

def save_model(model, optimizer, scheduler, cfg, time_string = ""):
    current_time = datetime.now()
    #time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    #if add_meta:
    name_file = f"protclip_{cfg.jobid}_{time_string}.pth"
    #else:    
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path, exist_ok=True)
    if cfg.save_every:
        PATH = os.path.join(cfg.save_path, name_file)
    else:
        PATH = os.path.join(cfg.save_path, name_file)
    print("saving model to......", PATH)
    if cfg.train.parallel == "dp":
        if scheduler is not None:
            torch.save({'scheduler_state_dict':scheduler.state_dict(), 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'cfg': cfg}, PATH)
        else:
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'cfg': cfg}, PATH)
    elif cfg.train.parallel == "ddp":
        if scheduler is not None:
            torch.save({'scheduler_state_dict':scheduler.state_dict(), 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'cfg': cfg}, PATH)
        else:
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'cfg': cfg}, PATH)
    print("saved!")


def generate_fasta_file(pid2seq_dict, save_path):
    with open(save_path, 'w') as f:
        for key in pid2seq_dict:
            f.write(">{}\n".format(key))
            f.write("{}\n".format(pid2seq_dict[key]))


def get_go_ancestors(graph, x):
    ancestors = set()
    ancestors.add(x)
    try:

        curr_ancestors = networkx.descendants(graph, x)
        ancestors.update(curr_ancestors)
    except:
        print("NOT FOUND IN ONTOLOGY: ", x)
    return list(ancestors)

def get_all_go_ancestors(graph, x):
    """
    graph: GO ontology graph
    x: set of GO labels for a given sequence
    """
    ancestors_and_labels = set()
    for label in x:
        ancestors_and_labels.update(get_go_ancestors(graph, label))
    return ancestors_and_labels

def get_go_graph(paths, return_data=True, return_path=False):
    file_path,_ = get_file_path_and_date_from_key("go-graph", paths)
    #print(file_path)
    #exit()
    go_graph = obonet.read_obo(file_path)
    if return_data:
        go_graph = go_graph.nodes(data=True)
    if return_path:
        return go_graph, file_path
    return go_graph




def contains_english_characters(input_string):
    for char in input_string:
        if char.isalpha():
            return True
    return False

def contains_number(s):
    # Use a regular expression to check for the presence of a number
    pattern = r'\d+'
    return bool(re.search(pattern, s))


from collections import Counter

def get_unique_occurrences(strings):
    # Count the occurrences of each string
    string_counter = Counter(strings)

    # Get unique occurrences and their frequencies
    unique_occurrences = list(string_counter.keys())
    frequencies = list(string_counter.values())

    return unique_occurrences, frequencies
class name_preprocessor():
    def __init__(self, fact_path):
        self.unique_last_token = []
        self.fact_path = fact_path
        all_last_tokens = []
        name_frame = compress_json.load(fact_path+"protein_recommended_name_frame.json.gz")
        fids = list(name_frame.keys())
        for ii, fid in enumerate(fids):
            key = "recommended_name"
            name = name_frame[fid]["content"][key]
            try:
                last_token = name.split()[-1]
                all_last_tokens += [last_token.lower()]
            except:
                continue
        last_token, frequency = get_unique_occurrences(all_last_tokens)
        #print(last_token, frequency)
        for ii in range(len(last_token)):
            if frequency[ii] == 1:
                self.unique_last_token += [last_token[ii]]
        self.unique_last_token = set(self.unique_last_token)


    def preprocess_name(self, name):
        new_name = name.lower()
        # remove if Has disallowed terms
        if new_name.startswith("uncharacterized protein"):
            return ""
        if new_name in ["na", "n/a"]:
            return ""
        disallowed_terms = ["uncharacterized", "genomic scaffold", "genomic, scaffold", "whole genome shotgun sequence",
                            '|']
        for d_t in disallowed_terms:
            if d_t == new_name:
                return ""

        # remove terms that indicate some level of uncertainty
        uncertain_keywords = ["putative", "probable", "low quality"]
        for ss in uncertain_keywords:
            if ss in new_name:
                new_name = new_name.replace(ss, "")


        # remove the last token that belongs to an ID system
        last_token = new_name.split()[-1]
        if "_" in last_token or last_token in self.unique_last_token:
            #print(last_token)
            new_name = " ".join(new_name.split()[:-1])


        # strip isform ending
        if "isoform" in new_name:
            prefix, _, suffix = new_name.partition("isoform")
            if suffix.strip().isdigit():
                new_name = prefix

        if new_name.isspace():
            return ""

        new_name = new_name.strip()
        # new_name = re.sub(r'\s+', ' ', new_name)
        new_name = " ".join(new_name.split())

        #if "chloroplastic" in new_name:
        #    print(new_name)
        if new_name == "protein":
            return ""

        return new_name


def preprocess_name(name):
    new_name = name.lower()
    #remove if Has disallowed terms
    if new_name.startswith("uncharacterized protein"):
        #print(new_name)
        return ""
    if new_name in ["na", "n/a"]:
        #print(new_name)
        return ""
    disallowed_terms = ["uncharacterized", "genomic scaffold", "genomic, scaffold", "whole genome shotgun sequence", '|']
    for d_t in disallowed_terms:
        if d_t == new_name:
            #print(new_name)
            return ""

    #remove terms that indicate some level of uncertainty
    uncertain_keywords = ["putative", "probable", "low quality"]
    for ss in uncertain_keywords:
        if ss in new_name:
            new_name = new_name.replace(ss, "")
            """
            print(new_name)
            print(name)
            print("----")
            """

    #remove the last token that belongs to an ID system
    last_token = new_name.split()[-1]
    if "_" in last_token:
        new_name = " ".join(new_name.split()[:-1])



    #strip isform ending
    if "isoform" in new_name:
        prefix, _, suffix = new_name.partition("isoform")
        if suffix.strip().isdigit():
            new_name = prefix
            """
            print(new_name)
            print(name)
            print("----")
            """

    if new_name.isspace():
        return ""

    new_name = new_name.strip()
    #new_name = re.sub(r'\s+', ' ', new_name)
    new_name = " ".join(new_name.split())


    if new_name == "protein":
        return ""
    """
    print(new_name)
    print(name)
    print("-----------")
    #print(new_name, name)
    """
    return new_name

def propagate_ancestor_prob(go, prob, graph, take_max_prob=True, remove_ancestor_probs=False):
    ## list of go_ids and probabilities where go_idx correopnds to prob_idx
    go2ancestors = {}
    ancestors = set()
    go_set = set(go)
    for i in range(len(go)):
        try:
            go2ancestors[go[i]] = networkx.descendants(graph, go[i])
        except:
            continue
        for ancestor in go2ancestors[go[i]]:
            if ancestor in go_set:
                ancestor_idx = go.index(ancestor)
                if remove_ancestor_probs:
                    prob[ancestor_idx] = 0
                elif take_max_prob:
                    prob[ancestor_idx] = max(prob[ancestor_idx], prob[i])
    return prob

def propagate_ancestor_prob_mtx(go, prob, graph, take_max_prob=True, remove_ancestor_probs=False):
    ## list of go_ids and probabilities where go_idx correopnds to prob_idx
    go2ancestors = {}
    ancestors = set()
    go_set = set(go)
    for i in range(len(go)):
        try:
            go2ancestors[go[i]] = networkx.descendants(graph, go[i])
        except:
            continue
        for ancestor in go2ancestors[go[i]]:
            if ancestor in go_set:
                ancestor_idx = go.index(ancestor)
                if remove_ancestor_probs:
                    prob[ancestor_idx] = 0
                elif take_max_prob:
                    prob[ancestor_idx] = max(prob[ancestor_idx], prob[i])
    return prob

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_go_annotations(go, real_annots, pred_annots,  orig_preds, orig_probs):
    total = 0
    p = 0.0
    r = 0.0
    wp = 0.0
    wr = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    avg_ic = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        tpic = 0.0
        for go_id in tp:
            tpic += go.get_norm_ic(go_id)
            avg_ic += go.get_ic(go_id)
        fpic = 0.0
        for go_id in fp:
            fpic += go.get_norm_ic(go_id)
            mi += go.get_ic(go_id)
        fnic = 0.0
        for go_id in fn:
            fnic += go.get_norm_ic(go_id)
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        wrecall = tpic / (tpic + fnic)
        wr += wrecall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
            wp += tpic / (tpic + fpic)
    avg_ic = (avg_ic + mi) / total
    ru /= total
    mi /= total
    r /= total
    wr /= total
    if p_total > 0:
        p /= p_total
        wp /= p_total
    f = 0.0
    wf = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
        wf = 2 * wp * wr / (wp + wr)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns, avg_ic, wf    

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def compute_go_metrics(go_rels, go2label, label2go, terms, eval_preds, labels, mlb, orig_preds, orig_probs, ont="cc",  save_path=None, t_incr=0.2):
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

    NAMESPACES = {
        'cc': 'cellular_component',
        'mf': 'molecular_function',
        'bp': 'biological_process'
    }
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])

    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    # Combine scores for diamond and deepgo
    alpha = 0.5
    fmax = 0.0
    tmax = 0.0
    wfmax = 0.0
    wtmax = 0.0
    avgic = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    threshold = 0

    #for t in range(0, 101):
    upper_t = int((1/t_incr) + 1)
    for t in range(0, upper_t):
        threshold = t * t_incr
        #threshold = t / 5.0
        preds = []
        if t == 0: continue
        result = np.array([np.where(row >= threshold, 1, 0) for row in eval_preds])
        preds = mlb.inverse_transform(result)

        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
        labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
        fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf = evaluate_go_annotations(go_rels, labels, preds, orig_preds, orig_probs)
        #fscore, prec, rec, s, ru, mi, fps, fns = evaluate_go_annotations(go_rels, labels, preds, orig_preds, orig_probs)
        precisions.append(prec)
        recalls.append(rec)

        output = f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}, WFmax: {wf}, AVG_IC: {avg_ic:.3f}'
#<<<<<<< HEAD
        #print(output)
#=======
        #output = f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}'

#print(output)
#>>>>>>> 1eb47b6650f807788a63f32e47a92cdd5f558603
        if save_path != None:
            with open(save_path, 'a') as f:
                f.write(output + '\n')

        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            avgic = avg_ic
        if wfmax < wf:
            wfmax = wf
            wtmax = threshold
        if smin > s:
            smin = s
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)

    best_results = f"Fmax: {fmax:0.3f}, AUPR: {aupr:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, WFmax: {wfmax:0.3f}, threshold: {wtmax}, AVGIC: {avgic:0.3f}"
    print(f'\n-----------------------\nFINAL RESULTS ONT: {ont}\n' + best_results + '\n-----------------------\n')
    #print("ONT", ont)
    #print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    #print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
    #print(f'AUPR: {aupr:0.3f}')
    #print(f'AVGIC: {avgic:0.3f}')
    if save_path != None:
        with open(save_path, 'a') as f:
            f.write(f'\n-----------------------\nFINAL RESULTS ONT: {ont}\n' + best_results + '\n-----------------------\n')




class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm =0

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id]) 
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        downloaded=False
        if "http" in filename:
            downloaded=True
            print("downloading go graph............", filename)
            filename = wget.download(filename)
            print("downloaded to:: ", filename)
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        if downloaded:
            os.remove(filename)
        return ont


    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

def eval_go(all_possible_labels, labels, prediction, ont, paths, save_path=None, propagate_score=True, keep_topk=True, t_incr=0.2, all_annotations=None):
#eval_go(terms, labels, prediction, ont, paths, args.save_path, propagate_score, keep_topk, t_incr=1/args.neighbours)
    """
    all_possible_labels: list of all possible labels that can be predicted
    target: List[List] where inner list is list of ground truth GO labels
    prediction: List[dict] where inner dict is {predicted_go_label: prob}
    """
    go_graph, go_path = get_go_graph(paths, return_data=False, return_path=True)
    #go_rels= Ontology("go-basic.obo", with_rels=True)
    go_rels= Ontology(go_path, with_rels=True)
    go_rels.calculate_ic(all_annotations)


    mlb = MultiLabelBinarizer()
    mlb.fit([all_possible_labels])
    terms = mlb.classes_
    go2label = {v: i for i, v in enumerate(terms)}
    label2go = {go2label[v]: v for v in go2label}

    preds, probs = [], []
    for ii in range(len(prediction)):
        preds_i, probs_i = [], []
        for key in prediction[ii]:
            preds_i.append(key)
            probs_i.append(prediction[ii][key])
        preds.append(preds_i)
        probs.append(probs_i)

    preds_mtx = np.array(mlb.transform(preds), dtype=np.float16)
    for ii in range(len(preds)):
        for jj in range(len(preds[ii])):
            col = go2label[preds[ii][jj]]
            preds_mtx[ii][col] = probs[ii][jj]
    if keep_topk:
        preds_mtx = keep_top_k(preds_mtx)
    if propagate_score:
        for ii in range(len(preds)):
            for jj in range(len(preds[ii])):
                col = go2label[preds[ii][jj]]
                if preds_mtx[ii][col] > 0:
                    ancestors = get_go_ancestors(go_graph, preds[ii][jj])
                    ancestor_cols = [go2label[a] for a in ancestors]
                    col = go2label[preds[ii][jj]]
                    for a in ancestor_cols:
                        preds_mtx[ii][a] = max(preds_mtx[ii][a], preds_mtx[ii][col])

    compute_go_metrics(go_rels, go2label, label2go, terms, preds_mtx, labels, mlb, preds, probs, ont=ont, save_path=save_path,t_incr=t_incr)

def keep_top_k(matrix, k=100):
    result = np.zeros_like(matrix)  # Create a zero matrix of the same shape
    for i, row in enumerate(matrix):
        top_k_indices = np.argsort(row)[-k:]  # Get indices of top 100 values
        result[i, top_k_indices] = row[top_k_indices]  # Set top 100 values
    return result

def exact_match(prediction, target):
    if prediction == target:
        return 1
    else:
        return 0

def substring_match(prediction, target):
    if prediction in target:
        return 1
    else:
        return 0

def bleu_score(prediction, target, chencherry):
    candidate_tokens = nltk.word_tokenize(prediction)
    if len(candidate_tokens) < 4:
        candidate_tokens += ["."] * (4 - len(candidate_tokens))
    reference_tokens = nltk.word_tokenize(target)
    if len(reference_tokens) < 4:
        reference_tokens += ["."] * (4 - len(reference_tokens))
    #print(candidate_tokens, reference_tokens)

    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=chencherry.method5)
    #print(prediction, target, bleu_score)
    return bleu_score


def compute_name_similarity(predictions, targets):
    assert len(predictions) == len(targets)
    exact_score = 0
    substring_score = 0
    bleu_scores = 0
    predictions = [ss.lower() for ss in predictions]
    targets = [ss.lower() for ss in targets]
    chencherry = SmoothingFunction()
    for ii in range(len(predictions)):
        exact_score += exact_match(predictions[ii], targets[ii])
        substring_score += substring_match(predictions[ii], targets[ii])
        bleu_scores += bleu_score(predictions[ii], targets[ii], chencherry)

    exact_score /= len(predictions)
    substring_score /= len(predictions)
    bleu_scores /= len(predictions)
    return exact_score, substring_score, bleu_scores


def go_name2term(go_graph):
    name2term = {}
    for go in go_graph:
        name2term[go[1]["name"]] = go[0]
    return name2term


def eval_go_fromfile(prediction_path, target_path):
    prediction_file = pd.read_csv(prediction_path, sep=" ", header=None)
    prediction_file = prediction_file.dropna()
    prediction_file.columns = ["pid", "go_term", "score"]
    prediction_file = prediction_file.groupby(['pid']).agg(tuple).applymap(list).reset_index()
    prediction = []
    pid = []
    for index, row in prediction_file.iterrows():
        pid += [row["pid"]]
        prediction += [dict(zip(row["go_term"], row["score"]))]
    #print(pid)
    target_file = pd.read_csv(target_path)
    target_file = target_file[target_file["pid"].isin(pid)]
    target_file["GO"] = target_file["GO"].apply(ast.literal_eval)
    #print(target_file["GO"])
    target = target_file["GO"].tolist()
    paths = yaml.safe_load(open("_config/paths.yml", 'r'))
    terms = []
    go_graph = get_go_graph(paths)
    #go_graph = obonet.read_obo(paths)
    for go in go_graph:
        #print(go)
        terms += [go[0]]
    #print(prediction)
    #print(target)
    #print(terms)
    #eval_go(terms, target, prediction, "cc", paths)
    #eval_go(terms, target, prediction, "mf", paths)
    eval_go(terms, target, prediction, "bp", paths)
    #paths = yaml.safe_load

def compute_f1(prediction, label):
    intersection_set = label & prediction
    if len(intersection_set) == 0:
        return 0
    precision = len(intersection_set) / len(prediction)
    recall = len(intersection_set) / len(label)
    return (2 * precision * recall) / (precision + recall)

def reformat_seq(x):
    if x[0] == "[":
        return ast.literal_eval(x)[0]
    else:
        return x

def reformat_label_list(x):
    if x != x: return None
    if x[0] == "[":
        return ast.literal_eval(x)
    else:
        return None

def random_crop_aaseq(s, max_length):
    assert len(s) > max_length
    #start_pos = 0
    start_pos = random.randint(0, len(s)-max_length)
    return s[start_pos:start_pos+max_length]

if __name__ == "__main__":
    prediction_path = "go_prediction_1201.txt"
    target_path = "/work1/maddison/haonand/val_set_mmseq10_uniref50_10per_name.csv"
    eval_go_fromfile(prediction_path, target_path)
    """
    predictions = ['tegument protein', 'protein phosphatase 1 regulatory', 'nicotinate-nucleotide pyrophosphorylase', '50s ribosomal protein 1234']
    targets = ['tegument protein', 'protein phosphatase 1 regulatory subunit', 'quinolinate phosphoribosyltransferase', '50s ribosomal protein']
    print(compute_name_similarity(predictions, targets))
    """

import argparse
import yaml
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, average_precision_score

import ast
import yaml
import pandas as pd
import itertools
import pickle
#from readouts import MLP
#import lightning as L
#from pytorch_lightning.loggers import WandbLogger
#from lightning.pytorch.callbacks.early_stopping import EarlyStopping
#from lightning.pytorch.callbacks import Callback,ModelCheckpoint
#from lightning.pytorch import seed_everything
#from datasets import PPIDataset, ppi_collate
from datasets import DownstreamDataset, ppi_collate
#import lightning.pytorch as pl
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
seed_everything(42, workers=True)

import wandb
#wandb.init(mode="disabled")

from get_best_hparams import get_best_configs


from tqdm import tqdm
import re
import os, sys, psutil
path_to_root = "/".join(os.getcwd().split("/")[:-3])
sys.path.append(path_to_root)
sys.path.insert(0,os.path.join(path_to_root, "biochem"))
from utils import * 

from evaluation.knn import FaissKNeighbors
from evaluation import utils

def get_eval_metrics(pred_label, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    # for including probability
    pred_m_auc = np.zeros((n_test, len(mlb.classes_)))
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
         # fill in probabilities for prediction
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, acc

def load_data(src_dir, df, task, label_col,pid_col='pid', load_embeds=True):
    pids, embeds, labels = [], [], []
    for ii, row in df.iterrows():
        if "PPI" in task:
            id1 = row['protein1_id']
            id2 = row['protein2_id']
            try:
                if load_embeds:
                    f = os.path.join(src_dir,"{}.pt".format(id1))
                    embed1 = torch.load(f).float().squeeze()
                    g = os.path.join(src_dir,"{}.pt".format(id2))
                    embed2 = torch.load(g).float().squeeze()
                    embeds.append([embed1, embed2])

                pids.append(f"{id1},{id2}")
                labels.append(row[label_col])
            except: continue

        else:
            pid = row[pid_col]
            if load_embeds:
                f = os.path.join(src_dir,"{}.pt".format(pid))
                embed = torch.load(f).float().squeeze()
                if "DPI" in task:
                    mol_id = row['molecule_id']
                    if "davis" in task:
                        f = os.path.join("/ssd005/projects/uniprot_aspuru/embeds_davis/GT4SD", "{}.pt".format(mol_id))
                    elif "BindingDB" in task:
                        f = os.path.join("/ssd005/projects/uniprot_aspuru/embeds_bindingdb/GT4SD", "{}.pt".format(mol_id))
                    mol_embed = torch.load(f).float().squeeze()
                    assert mol_embed.shape[0] == 768, mol_embed.shape
                    embed = torch.cat([mol_embed, embed])
                    pid = f"{mol_id},{pid}" 
                embeds.append(embed)
            pids.append(pid)
            labels.append(row[label_col])

    print("{} pids processed.".format(len(pids)))
    if "PPI" in task:
        return np.array(pids, dtype=object), embeds, labels
    else:
        return np.array(pids, dtype=object), np.array(embeds), labels
    #return np.array(pids, dtype=object), torch.stack(embeds), labels


def get_pid2label(train_pids, train_cafa_labels):
    return dict(zip(train_pids, train_cafa_labels))

def get_score(go_labels, normalize_by_dist=False, distances=None, neighbours=5):
    if not normalize_by_dist:
        """get count of cafa label occurrences and divide by max possible occurrences"""
        merged_labels = list(itertools.chain(*go_labels))
        labels, counts = np.unique(merged_labels, return_counts=True)
        counts = counts / neighbours
        return list(labels), counts
    else:
        """for every cafa label occurrence, sum distance and divide by total possible distance"""
        label2score = {}
        for i in range(len(go_labels)):
            dist = distances[i]
            for g in go_labels[i]:
                if g not in label2score:
                    label2score[g] = 0
                label2score[g] += dist
        dist_sum = sum(distances)
        labels, counts = zip(*label2score.items())
        counts = counts / dist_sum
    return labels, counts

def get_blast2seqidn(blast_reference):
    #blast2seqidn = {blast_reference[item][0]['Aligned_subject']: blast_reference[item][0]['Identities'] for item in blast_reference}
    blast2seqidn = {item: blast_reference[item][0]['Identities'] for item in blast_reference}
    data = []
    for key in blast2seqidn:
        data.append([key, blast2seqidn[key]])
    return data

def get_closest_blast(blast_dict, pid2label, pid, neighbours, args, blast_reference=None):
    nearest_pids = blast_dict[pid][:neighbours]
    if blast_reference != None:
        if pid not in blast_reference:  nearest_pids = []
        #blast_pids = blast_reference[pid][:neighbours]
        #blast_pids = blast_reference
        #if nearest_pids[0] not in blast_reference:
        #    nearest_pids = []
        #blast_pids = [item['Aligned_subject'] for item in blast_pids if item['Identities'] > args.seq_identity_min and item['Identities'] <= args.seq_identity_max]
        #if len(blast_pids) == 0:
        #    nearest_pids = []

        #nearest_pids [item['Aligned_subject'] for item in nearest_pids if item['Identities'] > seq_identity]
    #if len(nearest_pids) > 0 and type(nearest_pids[0]) == dict:
    #    nearest_pids = [item['Aligned_subject'] for item in nearest_pids if item['Identities'] > args.seq_identity_min and item['Identities'] <= args.seq_identity_max]
    sample_labels = []
    sample_pids = []
    for p in nearest_pids:
        if p in pid2label:
        #if p['Aligned_subject'] in pid2label:
            sample_labels.append(pid2label[p])
            sample_pids.append(p)

            #sample_labels.append(pid2label[p['Aligned_subject']])
            #sample_pids.append(p['Aligned_subject'])
        if len(sample_labels) == neighbours: break
    #if len(sample_labels) < neighbours: print("not enough neighbours!", pid); exit()
    #print(sample_labels, sample_pids)
    return sample_labels, sample_pids

def divide_into_bins(data):
    # Calculate the number of items per bin
    num_items_per_bin = max(len(data) // 20, 1)  # Ensure at least 1 item per bin
    
    # Sort the data by the second item in each sub-list
    sorted_data = sorted(data, key=lambda x: x[1])
    bins = np.array_split(sorted_data, 20)
    
    # Split the sorted data into bins
    #bins = [sorted_data[i:i + num_items_per_bin] for i in range(0, len(sorted_data), num_items_per_bin)]
    
    return bins

def get_knn(train_data, test_data, embeds_dir, args, task):
    neighbours = args.neighbours
    metric=args.metric
    weight_by_dist=False
    json_path=args.save_json_path

    train_pids, train_embeds, train_labels = train_data
    test_pids, test_embeds, test_labels = test_data

    if embeds_dir != None and  "json" in embeds_dir:
        assert os.path.exists(embeds_dir), f"if using JSON, model variable must be JSON dict: {embeds_dir}"
        print("loading precomputed dict.........")
        with open(embeds_dir) as f:
            knn_dict = json.load(f)
        
    
    else:
        print("Creading vector database........")
        vector_db = FaissKNeighbors(k=neighbours, metric=metric)
        print("Adding to vector database.......")
        vector_db.create_vector_db(X=train_embeds, pids=train_pids)
        print("Finding kNN.....................")
        distances, votes = vector_db.get_closest_embeds(X=test_embeds)
        if json_path != None:
            print("creating json....")
            val2train = {}
            for i in tqdm(range(len(test_pids))):
                for j in range(len(distances[i])-1):
                    assert distances[i][j] <= distances[i][j+1]

                val_pid = test_pids[i]
                closest_train_pids = votes[i]
                val2train[val_pid] = list(closest_train_pids)

            if embeds_dir[-1] == "/": embeds_dir = embeds_dir[:-1]
            #save_path = "/ssd005/projects/uniprot_aspuru/embeds/jsons/" + embeds_dir.split("/")[-1] + f"_{task}_{neighbours}NN.json"
            with open(json_path, "w") as f:
                print("saving to.........", json_path)
                json.dump(val2train, f)
            print("done saving dict!")

        assert len(test_pids) == len(votes)

    print("Computing final scores.........")
    pid2label = get_pid2label(train_pids, train_labels)
    pred_data = []
    pids, preds, probs, labels = [], [], [], []
    missing = 0
    blast_reference=None
    if args.mode ==  "knn_by_seq_identity":
        blast_ref_path=f"/ssd005/projects/uniprot_aspuru/blast/jsons/blast_alignment_mmseq10_uniref50_verbose_{task}.json"
        with open(blast_ref_path) as f:
            blast_reference = json.load(f)
        blast2seqidn = get_blast2seqidn(blast_reference)
        blast_bins = divide_into_bins(blast2seqidn)
        blast_pids = set()
        for item in blast_bins[args.seq_identity_bin]:
            blast_pids.add(item[0])
        print("len blast_pids", len(blast_pids))
        #    if i != 0: continue
        blast_reference=blast_pids

    for i in tqdm(range(len(test_pids))):
        pid = test_pids[i]
        if  "json" in embeds_dir:
            sample_labels, nearest_pids = get_closest_blast(knn_dict, pid2label, pid, neighbours, args, blast_reference) 
            if args.mode == "knn_by_seq_identity" and len(nearest_pids) == 0: continue
            else: assert len(nearest_pids) > 0
            dist = []
        else:
            nearest_pids = votes[i]
            dist = distances[i]
            sample_labels = [pid2label[p] for p in nearest_pids]
        pred, probability = get_score(sample_labels, normalize_by_dist=False, distances=dist, neighbours=neighbours)
        if i == 0:
            print("sample_labels", sample_labels)
            print("pred", pred)
            print("probability", probability)
            print("pid", pid)


        pids.append(pid)
        preds.append(pred)
        probs.append(probability)
        labels.append(test_labels[i])
    if True: #args.seq_identity_min > 0 or args.seq_identity_min < 100:
        print("--------------------------------------")
        print("filtering based on sequence similarity")
        print("before filtering....... ", len(test_pids))
        print("after filtering....... ", len(pids))
        print("--------------------------------------")
    return pids, preds, probs, labels 

def format_labels(x):
    if x == None:
        return None
    if type(x) == int: 
        x = str(x)
    if x[0] != "[":
        return [x]
    x = ast.literal_eval(x)
    if len(x) ==0: return None
    #if return_first_only:
    #    assert type(x) == list, "Danger! Trying to return first element of non-list type object"
    #    return x[0]
    return x

def load_df(path, label_col, subset, split_col=None, return_pid2cluster=False, task=None):
    df = pd.read_csv(path)

    #df['id'] = df['id'].astype("string")
    if split_col != None:
        df = df[df[split_col] == subset]
    df = df[df[label_col].notna()]
    if label_col == "BiDeeploc":
        df = df[df[label_col].isin(['M', 'S'])]
    if 'DPI' not in task:
        df[label_col] = df[label_col].apply(lambda x: format_labels(x))
    df = df[df[label_col].notna()]
    print(df)
    if return_pid2cluster:
        uniref_df = pd.read_csv("/ssd005/projects/uniprot_aspuru/datasets_alllen/val_set_mmseq10_uniref50_uniref.csv")
        pid2cluster = dict(zip(uniref_df.pid,uniref_df.uniref50))
        cluster2pid = dict(zip(uniref_df.uniref50,uniref_df.pid))
        return df, pid2cluster, cluster2pid
    return df

def combine_rows(x, y):
    return x + y

def get_weighted_score(scores, pids, pid2cluster, cluster2pid):
    cluster2scores = {}
    for i in range(len(pids)):
        pid = pids[i]
        score = scores[i]
        cluster_pid = pid2cluster[pid]
        if cluster_pid not in cluster2scores:
            cluster2scores[cluster_pid] = []
        cluster2scores[cluster_pid].append(score)
    weighted_score = 0
    for cluster in cluster2scores:
        weighted_score += np.mean(cluster2scores[cluster])
    weighted_score /= len(cluster2scores)
    return weighted_score

def run_go(pids, preds, probs, labels, args, propagate_score, keep_topk, paths, train_data, val_data, all_annotations):
    terms = []
    go_graph = get_go_graph(paths)
    for go in go_graph:
      terms += [go[0]]
    
    terms = set(terms)
    for p in train_data[-1]:
        terms.update(p)
    for p in val_data[-1]:
        terms.update(p)
    terms = list(terms)

    prediction = []
    for ii in range(len(preds)):
        dict_ = {}
        for jj in range(len(preds[ii])):
            dict_[preds[ii][jj]] = probs[ii][jj]
        prediction.append(dict_)

    for ont in ['cc', 'mf', 'bp']:
       print("------ RUNNING {} -----".format(ont))            
       eval_go(terms, labels, prediction, ont, paths, args.save_path, propagate_score, keep_topk, t_incr=1/args.neighbours, all_annotations=all_annotations)

#def eval_ec(train_data, val_data, args, pid2cluster, cluster2pid,paths=None):
def eval_ec(pids, preds, probs, labels, args, pid2cluster, cluster2pid, train_data, val_data):
    all_labels = set()
    for label in train_data[-1]:
        all_labels.update(label)
    for label in val_data[-1]:
        all_labels.update(label)
    all_labels = sorted(list(all_labels))

    #pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, args) 
    pre, rec, wf1, acc = get_eval_metrics(preds, labels, all_labels)
    if args.save_json_path == None:
        f1_score = 0
        scores = [] 
        for ii in range(len(preds)):
            f1 = compute_f1(set(preds[ii]), set(labels[ii]))
            f1_score += f1
            scores.append(f1)
        f1_score /= len(preds)
        print("Unweighted F1:: {}".format(f1_score))
        if pid2cluster != None:
            weighted_f1 = get_weighted_score(scores, pids, pid2cluster, cluster2pid) 
            print("Weighted F1 - by Uniref50:: {}".format(weighted_f1))
        print("Weighted F1 - by label:: {}".format(wf1))
        print("Precision:: {}".format(pre))
        print("Recall:: {}".format(rec))
        print("Accuracy:: {}".format(acc))
        if args.save_path != None:
            with open(args.save_path, 'a') as f:
                f.write(f"Num samples:: {len(pids)}\n")
                f.write(f"Unweighted F1 score:: {f1_score}\n")
                if pid2cluster != None:
                    f.write(f"Weighted F1 - by UniRef50 score:: {weighted_f1}\n")
                f.write(f"Weighted F1 - by label score:: {wf1}\n")
                f.write(f"Precision:: {pre}\n")
                f.write(f"Recall:: {rec}\n")
                f.write(f"Accuracy:: {acc}\n")

#def eval_peer(train_data, val_data, args, paths=None):
def eval_peer(pids, preds, probs, labels, args):

    #pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, args) 
    #print(pids[:10], preds[:10], probs[:10], labels[:10])
    #pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, neighbours=args.neighbours, metric=args.metric, save_json=args.save_json, task="names")
    acc = 0
    for ii in range(len(preds)):
        print("preds", preds[ii])
        print("probs", probs[ii])
        print("labels", labels[ii])
        maj_vote = np.argmax(probs[ii])
        print(maj_vote)
        pred = preds[ii][maj_vote]
        label = labels[ii][0]
        print("final pred", pred)
        print("final label", label)
        if pred == label:
            acc += 1
    acc /= len(preds)
    print('acc', acc)

#def eval_names(train_data, val_data, args, paths=None):
def eval_names(pids, preds, probs, labels, args):
    #pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, neighbours=args.neighbours, metric=args.metric, save_json=args.save_json, task="names")
    #pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, args) 
    if args.save_json_path == None:
        assert len(preds) == len(labels), (len(preds), len(labels))
        preds = [p[0] for p in preds]
        labels = [l[0] for l in labels]
        exact_score, substring_score, bleu_scores = compute_name_similarity(preds, labels)
        print("exact_score", exact_score)
        print("substring_score", substring_score)
        print("bleu_scores", bleu_scores)
        if args.save_path != None:
            with open(args.save_path, 'a') as f:
                f.write(f"Exact score:: {exact_score}\n")
                f.write(f"Substring score:: {substring_score}\n")
                f.write(f"BLEU score:: {bleu_scores}\n")

def write_log(save_path, command_run, args, train_data, val_data):
    if save_path != None:
        with open(save_path, 'a') as f:
            f.write(f"command executed: {command_run}\n\n")
            f.write(f"args: {args}\n\n")
            f.write(f"num_train: {len(train_data[0])}\n")
            f.write(f"num_val: {len(val_data[0])}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Script to run inference on Enzyme Commision dataset using our model or various baselines.")
    parser.add_argument('-i', '--idx', type=int, default=-1, help='job id if running in parallel')
    parser.add_argument('--few_shot', type=float, default=1)

    parser.add_argument('--seq_identity_min', type=int, default=0)
    parser.add_argument('--seq_identity_max', type=int, default=100)
    parser.add_argument('--seq_identity_bin', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('-n', '--neighbours',
                        default=1, type=int,
                        help='neighbours to vote over')
    parser.add_argument('-m', '--metric', default='l2', choices=['l2', 'cosine'], help='metric to use for distance computation')
    parser.add_argument('--mode', default='knn', choices=['knn', 'naive', 'mlp', "knn_by_seq_identity"], help='retrieval mode')
    parser.add_argument('--tasks', default='EC,names,spGO')
    parser.add_argument('--embeds_dir', type=str, help="path to embeds", default=None)
    parser.add_argument('--train_path', type=str, help="path to training set", default="/ssd005/projects/uniprot_aspuru/datasets_alllen/train_set_mmseq10_uniref50_spGO.csv")
    parser.add_argument('--val_path', type=str, help="path to val set", default="/ssd005/projects/uniprot_aspuru/datasets_alllen/val_set_mmseq10_uniref50_spGO.csv")
    parser.add_argument('--save_path', type=str, default=None, help="if not set to None, path where results will be saved")
    parser.add_argument('--save_json_path', default=None)
    parser.add_argument("--paths", default="/h/mskrt/ssl/mol_grasp/biochem/_config/paths.yml")
    parser.add_argument("--embedding_path", default=None)
    parser.add_argument("--embedding_path_test", default=None)
    parser.add_argument("--parse_test", action="store_true")

    # ARGS for LINEAR/MLP readout
    parser.add_argument("--lr", default=0, type=float)
    parser.add_argument("--batch_size", default=0, type=int)
    #parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--hidden_dim", default=0, type=int)
    

    command_run = psutil.Process(os.getpid())
    command_run = " ".join(command_run.cmdline())

    args = parser.parse_args()
    print(args)

    JSON_DIR = "jsons" 
    RESULTS_DIR = "results"
    
    if args.parse_test:
        JSON_DIR = "jsons_test"
        RESULTS_DIR = "results_test"
    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.embedding_path != None:
        os.makedirs(JSON_DIR, exist_ok=True)
        if args.embedding_path[-1] == "/": args.embedding_path = args.embedding_path[:-1]
        model_name = args.embedding_path.split("/")[-1]
        if args.embedding_path_test == None:
            args.embedding_path_test = args.embedding_path

    print(f"command executed: {command_run}\n")
    print(f"args: {args}\n")

    paths = yaml.safe_load(open(args.paths, 'r'))

    tasks = args.tasks.split(",")
    print("tasks::: ", tasks)
    #if args.save_json:
    for task in tasks: 
        print("on task.......... {}".format(task))
        task_in_saved_file = task
        if task in ['Deeploc', 'BiDeeploc']:
            task_in_saved_file = "deeploc"
        elif task in ['Deepsf']:
            task_in_saved_file = 'deepsf'
        elif task in ['Deepsol']:
            task_in_saved_file = 'deepsol'

        label_col=task
        pid_col='pid'
        split_col=None
        if task == 'spGO':
            label_col='GO'
        if task in ['BiDeeploc', 'Deeploc']:
            pid_col='UPID'
            split_col='test'
        if task in ['Deepsf']:
            pid_col='id'
            label_col='class'
            split_col='set'
        if task in ['Deepsol']:
            pid_col='id'
            label_col='class'
            split_col='set'
        if "PPI" in task: 
            label_col='interaction'
            split_col='split'
        if "DPI" in task:
            label_col='label'
            split_col='split'
            pid_col='target_id'


        pid2cluster, cluster2pid=None,None
        train_data = load_df(args.train_path, label_col, split_col=split_col, subset="train", task=task)
        if args.parse_test:
            #val_data = load_df(args.val_path, label_col,subset="val",split_col=split_col, return_pid2cluster=False, task=task)
            val_data, pid2cluster, cluster2pid = load_df(args.val_path, label_col,subset="test",  split_col=split_col,return_pid2cluster=True, task=task)
        else:
            val_data, pid2cluster, cluster2pid = load_df(args.val_path, label_col,subset="val",  split_col=split_col,return_pid2cluster=True, task=task)
        if args.few_shot != 1:
            assert args.few_shot < 1, "few shot must be frac"
            train_data = train_data.sample(frac=args.few_shot, replace=False, random_state=42)
            #val_data = train_data
        #train_data = train_data[:200]
        #val_data = val_data[:10]
        if task == "spGO":
            annotations = train_data[label_col].values
            annotations = list(map(lambda x: set(x), annotations))
            test_annotations = val_data[label_col].values
            test_annotations = list(map(lambda x: set(x), test_annotations))
            annotations = annotations + test_annotations
    
        if args.mode == "naive":
            print(train_data);
            print(val_data); 

            train_labels = train_data[label_col].to_list()
            avg_len = np.mean([len(l) for l in train_labels])

            if task == "spGO":
                num_to_retrieve = len(train_labels)
            else:
                num_to_retrieve = 1
            print("num_to_retrieve", num_to_retrieve)
            labels, counts = get_score(train_labels, neighbours=1)
            ordered_idx = np.argsort(counts)[::-1]
            print("max count", counts[ordered_idx[0]])
            #counts = counts/counts[ordered_idx[0]]
            counts = counts/len(train_labels)

            ordered_labels = [labels[i] for i in ordered_idx][:num_to_retrieve]
            frequency = [counts[i] for i in ordered_idx][:num_to_retrieve]
            print("frequency", frequency[:10])

            pids = val_data[pid_col].to_list()
            preds = [ordered_labels for _ in range(len(pids))]
            #probs = [[1]*len(ordered_labels) for _ in range(len(pids))]
            probs = [frequency for _ in range(len(pids))]
            labels = val_data[label_col].to_list()
            if not os.path.exists(f"{RESULTS_DIR}/{args.mode}"):
                os.makedirs(f"{RESULTS_DIR}/{args.mode}", exist_ok=True)
            if task =="spGO":
                args.neighbours=100

                args.save_path = f"{RESULTS_DIR}/{args.mode}/{task}_{args.neighbours}Fmax.txt"
                print("results path.....", args.save_path)
                print(f"args: {args}\n")
                write_log(args.save_path, command_run, args, [train_data], [val_data])

                run_go(pids, preds, probs, labels, args, propagate_score=False, keep_topk=False, paths=paths, train_data=[train_labels], val_data=[labels], all_annotations=annotations)
            else:
                args.neighbours=1
                args.save_path = f"{RESULTS_DIR}/{args.mode}/{task}.txt"
                print("results path.....", args.save_path)
                print(f"args: {args}\n")
                write_log(args.save_path, command_run, args, [train_data], [val_data])
                if task =="EC" or task in ["binding sites", "active sites", "sites", "pfam_domain"]:
                    eval_ec(pids, preds, probs, labels, args, pid2cluster, cluster2pid,train_data=[train_labels], val_data=[labels])
                elif task =="names" or task == "family":
                    eval_names(pids, preds, probs, labels, args)
                elif "deep" in task.lower():
                    eval_peer(pids, preds, probs, labels, args)
            continue

        if "protst" in args.embedding_path and "PPI" not in args.tasks:
            missing_protst_ids = []
            with open("missing_protst_ids.txt") as f:
                for line in f:
                    missing_protst_ids.append(line.replace(".pt", "")[:-1])
            train_data = train_data[~train_data[pid_col].isin(missing_protst_ids)]
            val_data = val_data[~val_data[pid_col].isin(missing_protst_ids)]
        
        if "knn" in args.mode:
            if "blast" in args.embedding_path:
                json_path = args.embedding_path + f"_{task_in_saved_file}.json"
            else:
                json_path = os.path.join(JSON_DIR, model_name + f"_{task_in_saved_file}_100NN.json")
            print("json_path", json_path)
            if not os.path.exists(json_path):
                print("first creating json file............")
                assert args.embedding_path != None, "need to first compute json from embeddings dir!"
                args.embeds_dir  = args.embedding_path
                args.save_json_path = json_path 
                args.neighbours = 100
                print("args", args)

                train_data = load_data(args.embedding_path, train_data, task, label_col=label_col, pid_col=pid_col, load_embeds=True)
                val_data = load_data(args.embedding_path_test, val_data, task, label_col=label_col,pid_col=pid_col, load_embeds=True) 
                
                get_knn(train_data, val_data, args.embeds_dir, args, task)
            else:
                train_data = load_data(None, train_data, task, label_col=label_col, pid_col=pid_col, load_embeds=False)
                val_data = load_data(None, val_data, task, label_col=label_col,pid_col=pid_col, load_embeds=False) 

            assert os.path.exists(json_path)
            args.embeds_dir = json_path
            args.save_json_path = None


            print("evaluating task.......", task)
            if task == "spGO":
                #neighbour_sweep=[1, 5,20,50]
                neighbour_sweep=[20]
                for n in neighbour_sweep:
                    args.neighbours=n
                    args.save_path = f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN/{model_name}_{args.neighbours}NN.txt"
                    print("results path.....", args.save_path)
                    if not os.path.exists(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN"):
                        os.makedirs(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN", exist_ok=True)

                    print(f"args: {args}\n")
                    write_log(args.save_path, command_run, args, train_data, val_data)

                    pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, args, task)
                    run_go(pids, preds, probs, labels, args, propagate_score=False, keep_topk=False, paths=paths, train_data=train_data, val_data=val_data, all_annotations=annotations)
            #elif task == "EC":
            elif task =="EC" or task in ["binding_sites", "active_sites", "sites", "pfam_domain"]:
                args.neighbours=1

                args.save_path = f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN/{model_name}_{args.neighbours}NN.txt"
                print("results path.....", args.save_path)
                if not os.path.exists(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN"):
                    os.makedirs(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN", exist_ok=True)

                print(f"args: {args}\n")
                write_log(args.save_path, command_run, args, train_data, val_data)

                pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, args, task)
                prediction = pd.DataFrame({"pid": pids, "predictions":preds, "labels":labels})
                print(f"writing to {RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN/results.csv")
                prediction.to_csv(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN/results_{model_name}.csv", index=False)
                exit()
                eval_ec(pids, preds, probs, labels, args, pid2cluster, cluster2pid,train_data, val_data)

            elif task == "names" or task == "family":
                args.neighbours=1
                args.save_path = f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN/{model_name}_{args.neighbours}NN.txt"
                print("results path.....", args.save_path)
                if not os.path.exists(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN"):
                    os.makedirs(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN", exist_ok=True)

                print(f"args: {args}\n")
                write_log(args.save_path, command_run, args, train_data, val_data)

                pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, args, task)
                eval_names(pids, preds, probs, labels, args)
            elif "deeploc" in task.lower() or "deepsf" in task.lower() or "deepsol" in task.lower(): 
                args.neighbours=20
                args.save_path = f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN/{model_name}_{args.neighbours}NN.txt"
                print("results path.....", args.save_path)
                if not os.path.exists(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN"):
                    os.makedirs(f"{RESULTS_DIR}/{args.mode}/{task}/{args.neighbours}NN", exist_ok=True)
                print(f"args: {args}\n")
                write_log(args.save_path, command_run, args, train_data, val_data)

                pids, preds, probs, labels = get_knn(train_data, val_data, args.embeds_dir, args, task)
                eval_peer(pids, preds, probs, labels, args)
        elif args.mode == "mlp":
            train_data = load_data(args.embedding_path, train_data, task, label_col=label_col, pid_col=pid_col, load_embeds=True)
            print("len train_data", len(train_data[0]))
            all_labels = set()

            val_data = load_data(args.embedding_path_test, val_data, task, label_col=label_col,pid_col=pid_col, load_embeds=True) 
            print("len val_data", len(val_data[0]))

            all_labels = set()
            if "DPI" not in task:
                unique_labels = set()
                for labels in val_data[-1]:
                    unique_labels = unique_labels.union(labels)
                for labels in train_data[-1]:
                    unique_labels = unique_labels.union(labels)
                unique_labels = sorted(list(unique_labels))
            loss = "CELoss"
            if task == "spGO":
                loss = "BCELoss"
            
            if not args.parse_test:
                exp_name = f"{task}_TRAIN_BESTLOSS_LINMULT_NEGSAMPLEv2_nl{args.num_layers}_{loss}_{model_name}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_hd{args.hidden_dim}"
                if args.few_shot < 1:
                    exp_name = f"{task}_TRAIN_BESTLOSS_LINMULT_NEGSAMPLEv2_nl{args.num_layers}_{loss}_{model_name}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_hd{args.hidden_dim}"
                    exp_name += f"_fs{args.few_shot}"
                label2idx = None
                if "DPI" not in task:
                    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
                    print("label2idx", label2idx)
            else: 
                best_checkpoint_path, best_run_config = get_best_configs(model_name, task) 
                best_checkpoint_path += "/model.ckpt"
                print("best_checkpoint_path", best_checkpoint_path)
            
            collate_fn = None

            if "PPI" in task:
                input_dim = train_data[1][0][0].shape[-1] * 2
                collate_fn = ppi_collate
            else:
                input_dim = train_data[1][0].shape[-1]

            if args.parse_test:
                model = MLP.load_from_checkpoint(checkpoint_path=best_checkpoint_path)#, num_classes=num_labels, args=args, task=task,
                print("LOADED MODEL from........", best_checkpoint_path)
                                 #input_dim=input_dim)#, label2idx=label2idx, idx2label=idx2label)
                print(model.label2idx)
                label2idx = model.label2idx
                #all_test = DownstreamDataset(val_data, "val", label2idx, task)
                trainer = L.Trainer()
                print("------------------------")
                #test_dataloader = torch.utils.data.DataLoader(all_test, batch_size=512, shuffle=False, num_workers=3, collate_fn=collate_fn)
                all_val = DownstreamDataset(val_data, "val", label2idx, task)
                val_dataloader = torch.utils.data.DataLoader(all_val, batch_size=512, shuffle=False, num_workers=3, collate_fn=collate_fn)

                # Evaluate model
                model.eval_label="test@bestloss"
                #model.eval_label="val@bestloss"
                trainer.test(model, dataloaders=val_dataloader)

                exit()

            else:
                hparams = {'lr': args.lr, 'batch_size': args.batch_size,
                        'num_layers': args.num_layers, 'weight_decay': args.weight_decay,
                        'hidden_dim': args.hidden_dim, 'patience': args.patience, 'few_shot': args.few_shot} 
                exp_info = {'task': task, 'model': model_name}
                all_train = DownstreamDataset(train_data, "train", label2idx, task)
                all_val = DownstreamDataset(val_data, "val", label2idx, task)

                train_dataloader = torch.utils.data.DataLoader(all_train, batch_size=hparams['batch_size'], shuffle=True, num_workers=3,collate_fn=None)
                val_dataloader = torch.utils.data.DataLoader(all_val, batch_size=512, shuffle=False, num_workers=3, collate_fn=collate_fn)


                wandb_logger = WandbLogger(log_model=True, project="prot_lm_readouts", 
                        name=exp_name, 
                        save_dir="/scratch/ssd004/scratch/mskrt/wandb_models/wandb",
                        tags=["val", task, model_name, f"{args.num_layers}_layers", "DEBUG"])

                wandb_logger.experiment.config.update(hparams)
                wandb_logger.experiment.config.update(exp_info)
                print("len train_data", len(train_data[0]))
                print("len val_data", len(val_data[0]))
                #print("num labels", len(label2idx))
                if "DPI" not in task and task != "spGO": 
                    labels, counts = np.unique(train_data[-1], return_counts=True)
                    print("train distr:::")
                    print(labels, counts)
                    labels, counts = np.unique(val_data[-1], return_counts=True)
                    print("val distr:::")
                    print(labels, counts)

                if task == "spGO":
                    early_stop_callback = EarlyStopping(monitor="val_loss_epoch", min_delta=0.00, patience=args.patience, verbose=False, mode="min")
                    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch", mode="min")
                    go_fn = {'paths': paths, 'train_data': train_data, 'val_data': val_data, 'annotations': annotations,
                        'fn': run_go}
                    model = MLP(num_labels, args, task=task, input_dim=input_dim, eval_fn=go_fn, label2idx=label2idx) #, idx2label=idx2label)
                else:
                    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=False, mode="min")
                    #checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
                    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
                    model = MLP(args, task=task, 
                                input_dim=input_dim, label2idx=label2idx) #, idx2label=idx2label)
                
                trainer = L.Trainer(max_epochs=1000, callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger, deterministic=True)
                trainer.fit(model, train_dataloader, val_dataloader)
                model.eval_label="val@bestloss"
                trainer.test(ckpt_path='best', dataloaders=val_dataloader)

                if task == "spGO":
                    trainer.test(ckpt_path='best', dataloaders=val_dataloader)
                    prediction = []
                    labels = []
                    pids = []
                    probs = []
                    for ii, batch in enumerate(val_dataloader):
                        p, _, y = batch
                        preds = model(batch)
                        for ii in range(len(preds)):
                            pids.append(p[ii])
                            preds_row = preds[ii].detach().numpy()
                            sorted_idx = np.argsort(preds_row)[::-1][:500]
                            pred_labels = [idx2label[s] for s in sorted_idx]
                            prediction.append(pred_labels)
                            scores = preds_row[sorted_idx]
                            probs.append(scores)

                            labels_row = []
                            for jj in range(len(y[ii])):
                                if y[ii][jj] == 1: labels_row.append(idx2label[jj])
                            labels.append(labels_row)
                    args.neighbours=100
                    args.save_path = exp_name + ".txt"
                    run_go(pids, prediction, probs, labels, args, False, False, paths, train_data, val_data, annotations)
                    exit()

            exit()
                
        
if __name__ == "__main__":
    main()


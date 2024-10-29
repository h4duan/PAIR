import argparse
import json
import pickle
import os
import compress_json

import yaml
import pandas as pd
import numpy as np
import sys
np.random.seed(42)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from _fact.protein_ec.batch import get_batch as get_ec_batch
from _fact.protein_recommended_name.batch import get_batch as get_name_batch
from _fact.protein_alternative_name.batch import get_batch as get_alt_name_batch
from _fact.protein_go.batch import get_batch as get_go_batch
from utils import * 
from get_test_pids import load_files, get_temporal_test_pids

def save_csv(val_pids, save_path, go_graph=None):
    pid2data = {pid: {"pid": pid} for pid in val_pids}
    fact_types = yaml.safe_load(open("../_config/fact_types.yml", 'r'))

    #paths = yaml.safe_load(open("../_config/paths_test.yml", 'r'))
    paths = yaml.safe_load(open("../_config/paths.yml", 'r'))
    pid_table = compress_json.load(paths["frames"]+"/id_table.json.gz")
    remove_pids = set(pid_table.keys()).difference(val_pids)
    #assert len(remove_pids) + len(val_pids) == len(pid_table)
    
    ## EC
    print("loading ec.........")
    args = fact_types["protein_ec"]
    args['joint'] = True
    frame = compress_json.load(paths["frames"]+"/protein_ec_frame.json.gz")
    val_ec, _ = get_ec_batch( frame, pid_table, None, len(val_pids), {"pids": remove_pids}, args , return_all_seq=True)
    for item in val_ec:
        assert item['pid'] in pid2data
        # 'text@1': 'This protein has the following EC numbers: 3.2.1.62, 3.2.1.108.'
        ec = item['text@1'][:-1].replace("This protein has the following EC numbers: ", "").split(", ")
        ec = [elem for elem in ec if len(elem.split(".")) == 4] # filter out EC numbers with less than 4 digits
        if len(ec) == 0: continue

        pid2data[item['pid']]['EC'] = ec
        pid2data[item['pid']]['seq'] = item['protein@1']

    ## RECOMMENDED NAME
    print("loading name.........")
    args = fact_types["protein_recommended_name"]
    frame = compress_json.load(paths["frames"]+"/protein_recommended_name_frame.json.gz")
    val_name, _ = get_name_batch( frame, pid_table, None, len(val_pids), {"pids": remove_pids}, args , return_all_seq=True)
    for item in val_name:
        #assert item['pid'] in pid2data
        # 'text@1': 'The name of the protein is Flagellar L-ring protein'
        name = [item['text@1'].replace("The name of the protein is ", "")]
        pid2data[item['pid']]['names'] = name
        pid2data[item['pid']]['seq'] = item['protein@1']

    ## ALTERNATIVE NAMES
    print("loading synonyms.........")
    args = fact_types["protein_alternative_name"]
    args['joint'] = True
    frame = compress_json.load(paths["frames"]+"/protein_alternative_name_frame.json.gz")
    val_alternative_names = get_alt_name_batch( frame, pid_table, None, len(val_pids), {"pids": remove_pids}, args , return_all_seq=True)
    for item in val_alternative_names:
        #assert item['pid'] in pid2data
        if 'names' not in pid2data[item['pid']]:
            pid2data[item['pid']]['names'] = []
            print(item)
        if "seq" not in pid2data[item['pid']]:
            pid2data[item['pid']]['seq'] = item['protein@1']
        #alt_names = item['text@1'].replace("Alternative names of this protein are: ", "").split(", ")
        alt_names = item['text@1'].replace("Alternative names of this protein are: ", "").split(" | ")
        pid2data[item['pid']]['names'].extend(alt_names)

    print("loading go.........")
    args = fact_types["protein_go"]
    args['joint'] = True
    args['filter_exp_tags'] = True
    args['p'] = 0
    #args['paths'] = yaml.safe_load(open("../_config/paths_test.yml", 'r'))
    args['paths'] = yaml.safe_load(open("../_config/paths.yml", 'r'))
    frame = compress_json.load(paths["frames"]+"/protein_go_frame.json.gz")
    remove_pids = set([frame[fid]["subjects"][0][4:] for fid in frame]).difference(val_pids)
    val_go = get_go_batch( frame, pid_table, None, len(val_pids), {"pids": remove_pids}, args , return_all_seq=True)
    for item in val_go:
        assert item['pid'] in pid2data
        #go = item['text@1'].replace("This protein has the following GO annotations: ", "").split(", ")
        go = item['text@1'].replace("This protein has the following GO annotations: ", "").split()
        go = ["GO:" + g for g in go]
        pid2data[item['pid']]['GO'] = set(go)
        if go_graph != None:
            for id_ in go:
                ancestors = get_go_ancestors(go_graph, id_)
                pid2data[item['pid']]['GO'].update(ancestors)
        pid2data[item['pid']]['GO'] = list(pid2data[item['pid']]['GO'])
        pid2data[item['pid']]['seq'] = item['protein@1']

    data = []
    for key in pid2data:
        if "seq" not in pid2data[key]: continue
        dep_pids = pid_table[key]['deprecated_pids']
        if dep_pids != None:
            pid2data[key]['deprecated_pids'] = dep_pids
        if len(pid2data[key]['seq'][0]) > 1024: continue
        data.append(pid2data[key])
        #if "names" not in pid2data[key]:
        #    print(pid2data[key])
    #print(val_df); exit()
    print("saving.........")
    val_df = pd.json_normalize(data)

    print(val_df)
    val_df.to_csv(save_path + ".csv")
    #generate_fasta_file({key: pid2data[key]['seq'][0] for key in pid2data}, save_path + ".fasta")
    generate_fasta_file({elem['pid']: elem['seq'][0] for elem in data}, save_path + ".fasta")

def get_pids(cluster_centers, pid_table, test_pids, mmseq_cluster2pids, uniref_cluster2pids):
    pids = set()
    for cluster in cluster_centers:
        uniref_cluster_centers = mmseq_cluster2pids[cluster]
        for c in uniref_cluster_centers:
            if c not in uniref_cluster2pids: print("missing cluster -- ", c); continue
            pids_to_add = uniref_cluster2pids[c]
            for p in pids_to_add:
                if p in pid_table:
                    assert p not in test_pids
                    pids.add(p)
    return pids
def get_validation_ids(cluster_path, val_percent=0.1, uniref_cluster2pids=None, pid_table=None, test_pids=None):
    cluster_df = pd.read_csv(cluster_path, sep="\t", names=['cluster_pid', 'pid'])
    cluster2pids = {}
    pid2cluster = {}
    for ii, row in cluster_df.iterrows():
        if row['cluster_pid'] not in cluster2pids:
            cluster2pids[row['cluster_pid']] = set()
        cluster2pids[row['cluster_pid']].add(row['pid'])
        pid2cluster[row['pid']] = row['cluster_pid']

    clusters = sorted(list(cluster2pids.keys()))

    print("there are {} clusters".format(len(clusters)))
    np.random.shuffle(clusters)
    val_size = int(len(clusters) * val_percent)
    val_clusters, train_clusters = clusters[:val_size], clusters[val_size:]
    print("there are {} val clusters and {} train clusters".format(len(val_clusters), len(train_clusters)))

    val_pids = set()
    for cluster in val_clusters:
        if cluster in pid_table:
            assert cluster not in test_pids
            val_pids.add(cluster)
        uniref_cluster_centers = cluster2pids[cluster]
        for c in uniref_cluster_centers:
            if c not in uniref_cluster2pids: print("missing val cluster -- ", c); continue
            pids_to_add = uniref_cluster2pids[c]
            for p in pids_to_add:
                if p in pid_table:
                    assert p not in test_pids
                    val_pids.add(p)
    train_pids = set()
    for cluster in train_clusters:
        if cluster in pid_table:
            assert cluster not in test_pids
            train_pids.add(cluster)
        uniref_cluster_centers = cluster2pids[cluster]
        for c in uniref_cluster_centers:
            if c not in uniref_cluster2pids: print("missing train cluster -- ", c); continue
            pids_to_add = uniref_cluster2pids[c]
            for p in pids_to_add:
                if p in pid_table:
                    assert p not in test_pids
                    train_pids.add(p)
            #train_pids.update(uniref50_cluster2pids[c])
    print("there are {} val seq and {} train seq".format(len(val_pids), len(train_pids)))
    return val_pids, train_pids

def generate_train_val_split(pid_table_train, uniref50_pid2cluster, uniref50_cluster2pid, test_pid_clusters, val_percent=0.1):
    clusters = set()
    for p in pid_table_train:
        if p in uniref50_pid2cluster:
            clusters.add(uniref50_pid2cluster[p])
        else:
            print("missing seq from uniref50.....", p)
    clusters = sorted(list(clusters))
    #clusters = [uniref50_pid2cluster[p] for p in pid_table_train]
    print("there are {} clusters before removing test".format(len(clusters)))
    clusters = [c for c in clusters if c not in test_pid_clusters]
    print("there are {} clusters after removing test".format(len(clusters)))
    clusters = sorted(clusters)
    #clusters = sorted(list(cluster2pids.keys()))

    print("there are {} clusters".format(len(clusters)))
    np.random.shuffle(clusters)
    val_size = int(len(clusters) * val_percent)
    val_clusters, train_clusters = clusters[:val_size], clusters[val_size:]
    assert len(set(train_clusters).difference(set(val_clusters))) == len(set(train_clusters))
    print("there are {} val clusters and {} train clusters".format(len(val_clusters), len(train_clusters)))
    val_pids = set()
    train_pids = set()

    for cluster in val_clusters:
        for pid in uniref50_cluster2pid[cluster]:
            if pid in pid_table_train: 
                val_pids.add(pid)
        #val_pids.update(uniref50_cluster2pid[cluster])
    for cluster in train_clusters:
        #train_pids.update(uniref50_cluster2pid[cluster])
        for pid in uniref50_cluster2pid[cluster]:
            if pid in pid_table_train: 
                train_pids.add(pid)

    print("num train pids:", len(train_pids))
    print("num val pids:", len(val_pids))
    return val_pids, train_pids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning Universal Representations for Biochemistry with Deep Neural Networks and Text Supervision"
    )
    #paths = yaml.safe_load(open("../_config/paths_test.yml", 'r'))
    paths = yaml.safe_load(open("../_config/paths.yml", 'r'))

    """ CONFIGURING RUN  """
    args = parser.parse_args()
    go_graph = get_go_graph(paths, return_data=False)

    paths_train = yaml.safe_load(open("../_config/paths.yml", 'r'))
    paths_test = yaml.safe_load(open("../_config/paths_test.yml", 'r'))

    #""" GETTING GLOBAL PID TABLE FROM UNIPROT  """
    assert os.path.exists(paths_train["frames"]+"/id_table.json.gz"), "swissprot from 202302 not processed!"

    pid_table_train, pid_table_test, uniref50_pid2cluster, uniref50_cluster2pid = load_files(paths_train, paths_test)
    test_pids = get_temporal_test_pids(pid_table_train, pid_table_test, uniref50_pid2cluster)
    all_test_pids = expand_test_seqs(test_pids, uniref50_cluster2pid, uniref50_pid2cluster)
    test_pid_centers = set()
    for pid in all_test_pids:
        test_pid_centers.add(uniref50_pid2cluster[pid])
    for path in paths:
        if "mmseq2-split10-uniref50" in path: cluster_path = paths[path]
    #cluster_path = "mmseq2_split10_uniref100_cluster.tsv"
    with open("uniref1002pid.json") as f:
        uniref1002pid = json.load(f)
    val_pids, train_pids = get_validation_ids(cluster_path,val_percent=0.10,
                                              uniref_cluster2pids=uniref1002pid, pid_table=pid_table_train,
                                              test_pids=all_test_pids) 
    with open("val_set_mmseq10_uniref100.txt", 'w') as f:
        for pid in val_pids:
            f.write(pid + '\n')
    with open("train_set_mmseq10_uniref100.txt", 'w') as f:
        for pid in train_pids:
            f.write(pid + '\n')

    save_csv(val_pids, save_path="/ssd005/projects/uniprot_aspuru/validation_sets/val_set_mmseq10_uniref100_10per", go_graph=go_graph)
    save_csv(train_pids, save_path="/ssd005/projects/uniprot_aspuru/validation_sets/train_set_mmseq10_uniref100_90per", go_graph=go_graph)

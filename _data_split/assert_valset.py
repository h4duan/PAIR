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


def get_validation_ids(cluster_path, val_percent=0.1, uniref50_cluster2pids=None, pid_table=None):
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
        #val_pids.add(cluster)
        uniref_cluster_centers = cluster2pids[cluster]
        #val_pids.update(uniref_cluster_centers)
        for c in uniref_cluster_centers:
            if c not in uniref50_cluster2pids:
                print("missing val cluster -- ", c)
                continue
            pids_to_add = uniref50_cluster2pids[c]
            for p in pids_to_add:
                if p in pid_table:
                    val_pids.add(p)
    train_pids = set()
    for cluster in train_clusters:
        train_pids.add(cluster)
        #train_pids.update(cluster2pids[cluster])
        uniref_cluster_centers = cluster2pids[cluster]
        train_pids.update(uniref_cluster_centers)
        for c in uniref_cluster_centers:
            if c not in uniref50_cluster2pids:
                print("missing train cluster -- ", c)
                continue
            pids_to_add = uniref50_cluster2pids[c]
            for p in pids_to_add:
                if p in pid_table:
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
    parser.add_argument('--paths_train', required=True)
    parser.add_argument('--paths_test', required=True)

    parser.add_argument('--path_to_test_cluster_pids', type=str, default=None)
    parser.add_argument('--additional_test_path', type=str, default=None)

    parser.add_argument('--val_path', type=str, default=None)
    #parser.add_argument('--train_path', type=str, default=None)

    parser.add_argument('--split_from_scratch', action="store_true")
    args = parser.parse_args()

    #if not args.split_from_scratch:
    #    assert args.val_path != None

    paths_train = yaml.safe_load(open(args.paths_train, 'r'))
    paths_test = yaml.safe_load(open(args.paths_test, 'r'))

    """ CONFIGURING RUN  """
    go_graph = get_go_graph(paths_train, return_data=False)

    ##""" GETTING GLOBAL PID TABLE FROM UNIPROT  """
    assert os.path.exists(paths_train["frames"]+"/id_table.json.gz"), "swissprot from 202302 not processed!"
    pid_table_train, pid_table_test, uniref50_pid2cluster, uniref50_cluster2pid = load_files(paths_train, paths_test, load_uniref=True)
    val_pids = pd.read_csv("/ssd005/projects/uniprot_aspuru/datasets/val_set_mmseq10_uniref50.csv")['pid'].to_numpy()
    val_pids = set(val_pids)

    sp_test_pids = set()
    peer_test_pids = set()
    with open(args.path_to_test_cluster_pids) as f:
        for line in f:
            sp_test_pids.add(line[:-1])

    #with open(args.additional_test_path) as f:
    #    for line in f:
    #        peer_test_pids.add(line[:-1])

    with open("/ssd005/projects/uniprot_aspuru/haonan_pid.pickle", 'rb') as f:
        d = pickle.load(f)

    train_pids = list(d.keys())
    #train_pids = [pid for pid in pid_table_train if pid not in val_pids]
    #train_pids = [p for p in train_pids if p not in peer_test_pids]
    #train_pids = [p for p in train_pids if p not in sp_test_pids]

    val_and_dep = set()
    for pid in val_pids:
        val_and_dep.add(pid)
        if pid_table_train[pid]['deprecated_pids']:
            for p in pid_table_train[pid]["deprecated_pids"]:
                val_and_dep.add(p)

    sp_test_and_dep = set()
    for pid in sp_test_pids:
        sp_test_and_dep.add(pid)
        if pid in pid_table_train and pid_table_train[pid]['deprecated_pids']:
            for p in pid_table_train[pid]["deprecated_pids"]:
                sp_test_and_dep.add(p)
        if pid in pid_table_test and pid_table_test[pid]['deprecated_pids']:
            for p in pid_table_test[pid]["deprecated_pids"]:
                sp_test_and_dep.add(p)
    
    peer_test_and_dep = set()
    for pid in peer_test_pids:
        peer_test_and_dep.add(pid)
        if pid in pid_table_train and pid_table_train[pid]['deprecated_pids']:
            for p in pid_table_train[pid]["deprecated_pids"]:
                peer_test_and_dep.add(p)
        if pid in pid_table_test and pid_table_test[pid]['deprecated_pids']:
            for p in pid_table_test[pid]["deprecated_pids"]:
                peer_test_and_dep.add(p)

    val_and_cluster = set()
    for pid in val_and_dep:
        val_and_cluster.add(pid)
        if pid in uniref50_pid2cluster:
            cluster_center = uniref50_pid2cluster[pid]
            cluster_pids = uniref50_cluster2pid[cluster_center]
            for p_ in cluster_pids:
                if p_ in pid_table_train:
                    val_and_cluster.add(p_)

            
    train_and_cluster = set()
    for pid in train_pids:
        train_and_cluster.add(pid)
        if pid in uniref50_pid2cluster:
            cluster_center = uniref50_pid2cluster[pid]
            cluster_pids = uniref50_cluster2pid[cluster_center]
            for p_ in cluster_pids:
                if p_ in pid_table_train:
                    train_and_cluster.add(p_)

    test_and_cluster = set()
    for pid in sp_test_and_dep:
        test_and_cluster.add(pid)
        if pid in uniref50_pid2cluster:
            cluster_center = uniref50_pid2cluster[pid]
            cluster_pids = uniref50_cluster2pid[cluster_center]
            for p_ in cluster_pids:
                if p_ in pid_table_train:
                    test_and_cluster.add(p_)
                    

    overlap = set(val_and_cluster) & set(train_and_cluster)
    print("overlap between trian and val:::", len(overlap))
    with open("overlap_val_train.txt", 'w') as f:
        for pid in overlap:
            f.write(f"{pid}\n")

    overlap = set(val_and_cluster) & set(test_and_cluster)
    print("overlap between val and sp test:::", len(overlap))
    with open("overlap_val_sptest.txt", 'w') as f:
        for pid in overlap:
            f.write(f"{pid}\n")

    overlap = set(train_and_cluster) & set(test_and_cluster)
    print("overlap between train and sp test:::", len(overlap))
    with open("overlap_train_sptest.txt", 'w') as f:
        for pid in overlap:
            f.write(f"{pid}\n")

    #overlap = set(val_and_cluster) & set(peer_test_and_dep)
    #print("overlap between val and peer test:::", len(overlap))
    #with open("overlap_val_peertest.txt", 'w') as f:
    #    for pid in overlap:
    #        f.write(f"{pid}\n")

    #overlap = set(train_and_cluster) & set(peer_test_and_dep)
    #print("overlap between train and peer test:::", len(overlap))
    #with open("overlap_train_peertest.txt", 'w') as f:
    #    for pid in overlap:
    #        f.write(f"{pid}\n")



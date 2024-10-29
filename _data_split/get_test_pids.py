import argparse
import yaml
import compress_json
import json
import path
import sys, os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils import expand_test_seqs
from torch.utils.data import DataLoader
import os
import wandb
from torch import optim

def get_temporal_test_pids(pid_table_train, pid_table_test, uniref50_pid2cluster=None):
    old_pids = set()
    new_pids = {}
    for pid in pid_table_train:
        dep_pids = pid_table_train[pid]['deprecated_pids']
        old_pids.add(pid)
        if dep_pids != None:
            assert type(dep_pids) == list
            old_pids.update(dep_pids)
    for pid in pid_table_test:
        dep_pids = pid_table_test[pid]['deprecated_pids']
        new_pids[pid] = set()
        new_pids[pid].add(pid)
        if dep_pids != None:
            assert type(dep_pids) == list
            new_pids[pid].update(dep_pids)
        
    test_pids = set()
    current_and_depracated_test_pids = set()

    for pid in new_pids:
        seen = False
        for key in new_pids[pid]:
            if key in old_pids:
                seen = True
        if not seen: #and pid in uniref50_pid2cluster:
            test_pids.add(pid)
            current_and_depracated_test_pids.add(pid)
            current_and_depracated_test_pids.update(new_pids[pid])

    print("{} test pids found".format(len(test_pids)))
    print("length of current_and_depracated_test_pids", len(current_and_depracated_test_pids))
    return test_pids, current_and_depracated_test_pids


def filter_train_pids(pid_table_train, test_pids, uniref_cluster2pid, uniref_pid2cluster):
    pids_to_remove = expand_test_seqs(test_pids, uniref_cluster2pid, uniref_pid2cluster)
    print("there are {} sequences in SP and TremBL in clusters that contain test pids. removing from training now!".format(len(pids_to_remove)))
    print("num train sequences before filtering:::", len(pid_table_train))
    new_train_dict = {}
    for pid in pid_table_train:
        pids_to_consider = set([pid])
        assert(len(pids_to_consider)) == 1
        if pid_table_train[pid]['deprecated_pids'] != None:
            assert type(pid_table_train[pid]['deprecated_pids']) == list
            pids_to_consider.update(pid_table_train[pid]['deprecated_pids']) 
        pid_in_remove_list = False
        for elem in pids_to_consider:
            if elem in pids_to_remove:
                pid_in_remove_list = True
        if not pid_in_remove_list:
            new_train_dict[pid] = pid_table_train[pid]
    print("num train sequences after filtering:::", len(new_train_dict))
    return new_train_dict, pids_to_remove

def generate_fasta_file(new_train_dict):
    with open("sp_trainval_202302_testremoved.fasta", 'w') as f:
        for key in new_train_dict:
            f.write(">{}\n".format(key))
            f.write("{}\n".format(new_train_dict[key]['aaseq']))

def load_files(paths_train, paths_test, load_uniref=True):
    pid_table_train = compress_json.load(paths_train["frames"]+"/id_table.json.gz")
    print("num entries in 2023-02::", len(pid_table_train))
    pid_table_test = compress_json.load(paths_test["frames"]+"/id_table.json.gz")
    #pid_table_test = compress_json.load(paths_test["frames"]+"/id_table_202309.json.gz")
    print("num entries in 2023-11::", len(pid_table_test))
    
    # load uniref50: assert all test pids are here, else don't include
    if load_uniref:
        uniref50_pid2cluster_path = None
        for path in paths_train:
            if "uniref50-pid2cluster" in path: uniref50_pid2cluster_path = paths_train[path]
        assert uniref50_pid2cluster_path != None, "must include path to uniref50_pid2cluster!"
        with open(uniref50_pid2cluster_path) as f:
            uniref50_pid2cluster = json.load(f)

        uniref50_cluster2pid_path = None
        for path in paths_train:
            if "uniref50-cluster2pid" in path: uniref50_cluster2pid_path = paths_train[path]
        assert uniref50_cluster2pid_path != None, "must include path to uniref50_cluster2pid!"
        with open(uniref50_cluster2pid_path) as f:
            uniref50_cluster2pid = json.load(f)
    else:
        uniref50_pid2cluster, uniref50_cluster2pid = None, None 

    return pid_table_train, pid_table_test, uniref50_pid2cluster, uniref50_cluster2pid 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Universal Representations for Biochemistry with Deep Neural Networks and Text Supervision")
    parser.add_argument('--paths_train', required=True)
    parser.add_argument('--paths_test', required=True)

    """ CONFIGURING RUN  """
    args = parser.parse_args()
    paths_train = yaml.safe_load(open(args.paths_train, 'r'))
    paths_test = yaml.safe_load(open(args.paths_test, 'r'))

    """ GETTING GLOBAL PID TABLE FROM UNIPROT  """
    assert os.path.exists(paths_train["frames"]+"/id_table.json.gz"), "swissprot from 202302 not processed!"

    pid_table_train, pid_table_test, uniref50_pid2cluster, uniref50_cluster2pid = load_files(paths_train, paths_test)
    #test_pids, current_and_depracated_test_pids = get_temporal_test_pids(pid_table_train, pid_table_test, uniref50_pid2cluster)
    test_pids, current_and_depracated_test_pids = get_temporal_test_pids(pid_table_train, pid_table_test)
    #filtered_train_pid_dict = filter_train_pids(pid_table_train, test_pids, uniref50_cluster2pid, uniref50_pid2cluster)
    filtered_train_pid_dict, pids_to_remove = filter_train_pids(pid_table_train, current_and_depracated_test_pids, uniref50_cluster2pid, uniref50_pid2cluster)

    with open("test_202401_not_in_uniref50.txt", 'w') as f:
        for pid in current_and_depracated_test_pids:
            if pid not in uniref50_pid2cluster:
                f.write(f"{pid}\n")

    with open("/ssd005/projects/uniprot_aspuru/pids/test_202401_uniref50_clusters.txt", 'w') as f:
        for pid in pids_to_remove:
            f.write(f"{pid}\n")

    with open("sp_trainval_202302_202401testremoved.json", 'w') as f:
        json.dump(filtered_train_pid_dict, f)
    with open("/ssd005/projects/uniprot_aspuru/pids/sp_trainval_202302_202401testremoved.txt", 'w') as f:
        for pid in filtered_train_pid_dict:
            f.write(f"{pid}\n")
    with open("/ssd005/projects/uniprot_aspuru/pids/sp_test_202401_test.txt", 'w') as f:
        for pid in current_and_depracated_test_pids:
            f.write(f"{pid}\n")

    generate_fasta_file(filtered_train_pid_dict)

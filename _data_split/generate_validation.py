import pickle
import argparse
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
from _fact.protein_domain.batch import get_batch as get_domain_batch
from _fact.protein_similarity_text.batch import get_batch as get_similarity_text_batch
from _fact.protein_binding_sites.batch import get_batch as get_binding_sites_batch
from _fact.protein_active_sites.batch import get_batch as get_active_sites_batch
from _fact.protein_sites.batch import get_batch as get_sites_batch
from utils import * 
from get_test_pids import load_files, get_temporal_test_pids

def save_csv(val_pids, save_path, paths, pid_table, go_graph=None, include_go_anc=True):
    pid2data = {pid: {"pid": pid} for pid in val_pids}
    for key in pid_table:
        pid_table[key]['uniref50'] = []
    for pid in pid2data:
        pid2data[pid]['seq'] = [pid_table[pid]['aaseq']]

    fact_types = yaml.safe_load(open("../_config/fact_types.yml", 'r'))

    #paths = yaml.safe_load(open("../_config/paths_test.yml", 'r'))
    #paths = yaml.safe_load(open("../_config/paths.yml", 'r'))
    print("PATHS", paths)
    #pid_table = compress_json.load(paths["frames"]+"/id_table.json.gz")
    remove_pids = set(pid_table.keys()).difference(val_pids)
    #remove_pids = set()
    #assert len(remove_pids) + len(val_pids) == len(pid_table)

    ## DOMAINS
    print("loading domains.............")
    args = fact_types['protein_domain']
    args['paths'] = paths
    args['parse_text'] = False
    args['parse_interpro'] = False
    args['parse_pfam'] = True
    args['joint'] = True
    frame = compress_json.load(paths["frames"]+"/protein_domain_frame.json.gz")
    fids = [key for key in frame if frame[key]['subjects'][0][4:] in pid_table]
    val_domains, _ = get_domain_batch( frame, fids, pid_table, None, len(val_pids), {"pids": remove_pids}, args)
    # frame, fids, pid_table, cid_table, size, id_filter, args
    print("num domains", len(val_domains)); 
    for item in val_domains:
        if item['pid'] not in pid2data: continue
        domain = item['text@1'].replace("Domain: ", "")
        if 'pfam_domain' not in pid2data[item['pid']]:
            pid2data[item['pid']]['pfam_domain'] = []
        pid2data[item['pid']]['pfam_domain'].append(domain.lower())

    ## FAMILY
    print("loading family.............")
    args = fact_types['protein_family']
    frame = compress_json.load(paths["frames"]+"/protein_family_frame.json.gz")
    fids = [key for key in frame if frame[key]['subjects'][0][4:] in pid_table]
    val_families, _ = get_similarity_text_batch( frame, fids,  pid_table, None, len(val_pids), {"pids": remove_pids}, args)
    # frame, fids, pid_table, cid_table, size, id_filter, args
    print("num familes", len(val_families)); 
    for item in val_families:
        if item['pid'] not in pid2data: continue
        family = item['text@1']
        # 'Belongs to the asfivirus MGF 110 family.'

        if 'family' not in pid2data[item['pid']]:
            pid2data[item['pid']]['family'] = []
        pid2data[item['pid']]['family'].append(family.lower())

    ## BINDING SITES
    print("loading binding sites.............")
    args = fact_types['protein_binding_sites']
    frame = compress_json.load(paths["frames"]+"/protein_binding_sites_frame.json.gz")
    fids = [key for key in frame if frame[key]['subjects'][0][4:] in pid_table]
    val_binding_sites, _ = get_binding_sites_batch( frame,  fids, pid_table, None, len(val_pids), {"pids": remove_pids}, args)
    # frame, fids, pid_table, cid_table, size, id_filter, args
    print("num bs", len(val_binding_sites)); 
    for item in val_binding_sites:
        if item['pid'] not in pid2data: continue
        binding_sites = item['text@1'].replace("Binding sites: ", "")
        binding_sites = ' '.join(binding_sites.split())
        if 'binding_sites' not in pid2data[item['pid']]:
            pid2data[item['pid']]['binding_sites'] = []
        pid2data[item['pid']]['binding_sites'].append(binding_sites.lower())


    ## ACTIVE SITES
    print("loading active sites.............")
    args = fact_types['protein_active_sites']
    frame = compress_json.load(paths["frames"]+"/protein_active_sites_frame.json.gz")
    fids = [key for key in frame if frame[key]['subjects'][0][4:] in pid_table]
    val_active_sites, _ = get_active_sites_batch( frame,  fids, pid_table, None, len(val_pids), {"pids": remove_pids}, args)
    # frame, fids, pid_table, cid_table, size, id_filter, args
    print("num active_sites", len(val_active_sites)); 
    for item in val_active_sites:
        if item['pid'] not in pid2data: continue
        active_sites = item['text@1'].replace("Active site: ", "").replace(".", "" )
        active_sites = ' '.join(active_sites.split())
        if 'active_sites' not in pid2data[item['pid']]:
            pid2data[item['pid']]['active_sites'] = []
        pid2data[item['pid']]['active_sites'].append(active_sites.lower())

    ## SITES
    print("loading active sites.............")
    args = fact_types['protein_sites']
    frame = compress_json.load(paths["frames"]+"/protein_sites_frame.json.gz")
    fids = [key for key in frame if frame[key]['subjects'][0][4:] in pid_table]
    val_sites, _ = get_sites_batch( frame,  fids, pid_table, None, len(val_pids), {"pids": remove_pids}, args)
    # frame, fids, pid_table, cid_table, size, id_filter, args
    print("num sites", len(val_sites)); 
    for item in val_sites:
        if item['pid'] not in pid2data: continue
        sites = item['text@1'].replace("Sites: ", "").replace(".", "" )
        sites = ' '.join(sites.split())
        if 'sites' not in pid2data[item['pid']]:
            pid2data[item['pid']]['sites'] = []
        pid2data[item['pid']]['sites'].append(sites.lower())
    
    ## EC
    print("loading ec.........")
    args = fact_types["protein_ec"]
    args['joint'] = True
    frame = compress_json.load(paths["frames"]+"/protein_ec_frame.json.gz")
    val_ec, _ = get_ec_batch( frame, pid_table, None, len(val_pids), {"pids": remove_pids}, args , return_all_seq=True)
    print("num ec", len(val_ec))
    for item in val_ec:
        if item['pid'] not in pid2data: continue
        ec = item['text@1'][:-1].replace("This protein has the EC number: ", "").split(", ")
        ec = [".".join(elem.split()) for elem in ec]
        ec = [elem for elem in ec if len(elem.split(".")) == 4 and 'n' not in elem] # filter out EC numbers with less than 4 digits
        if len(ec) == 0: continue
        pid2data[item['pid']]['EC'] = ec
    print("num ec", len(pid2data))

    ## RECOMMENDED NAME
    print("loading name.........")
    args = fact_types["protein_recommended_name"]
    args['preprocess'] = False
    name_processor=None
    #if args['preprocess']:
    #    name_processor = name_preprocessor("/ssd005/projects/uniprot_aspuru/biochem_frames/")
    frame = compress_json.load(paths["frames"]+"/protein_recommended_name_frame.json.gz")
    fids = [key for key in frame if frame[key]['subjects'][0][4:] in pid_table]
    val_name, _ = get_name_batch( frame,fids, pid_table, None, len(val_pids), {"pids": remove_pids}, args , name_processor=name_processor, return_all_seq=True)
    print("num name", len(val_name))
    for item in val_name:
        if item['pid'] not in pid2data: continue
        #name = item['text@1'].replace("The name of the protein is ", "")
        # Recommended name: Protein MGF 110-6L.
        name = item['text@1']

        if name[-1] == ".":
            name = name[:-1]
        name = [name]
        assert 'names' not in pid2data[item['pid']], item
        pid2data[item['pid']]['names'] = name
        #pid2data[item['pid']]['seq'] = item['protein@1']


    print("loading go.........")
    args = fact_types["protein_go"]
    args['joint'] = True
    args['filter_exp_tags'] = True
    args['p'] = 0
    args['incl_ancestors'] = True
    args['paths'] = yaml.safe_load(open("../_config/paths.yml", 'r'))
    frame = compress_json.load(paths["frames"]+"/protein_go_frame.json.gz")
    #remove_pids = set([frame[fid]["subjects"][0][4:] for fid in frame]).difference(val_pids)
    val_go, _ = get_go_batch( frame, pid_table, None, len(val_pids), {"pids": remove_pids}, args , return_all_seq=True)
    print("num go", len(val_go))
    for item in val_go:
        if item['pid'] not in pid2data: continue
        go = item['text@1'][:-1].replace("This protein has the GO annotations: ", "").split()
        go = ["GO:" + g.replace('"', "") for g in go]
        pid2data[item['pid']]['GO'] = list(go)

    #names = []
    #for name in pid2data[item['pid']]['names']:
    #    if name != "":
    #        names.append(name)
    #print(names[:10]); exit()
    #if len(names) > 0:
    #    pid2data[item['pid']]['names'] = names
    #else:
    #    pid2data[item['pid']]['names'] = None

    data = []
    for key in pid2data:
        dep_pids = pid_table[key]['deprecated_pids']
        if dep_pids != None:
            pid2data[key]['deprecated_pids'] = dep_pids
        data.append(pid2data[key])
    print("saving.........")
    val_df = pd.json_normalize(data)

    print(val_df)
    val_df.to_csv(save_path + ".csv")
    generate_fasta_file({elem['pid']: elem['seq'][0] for elem in data}, save_path  + ".fasta")
    for task in ["EC", "GO", "names", 'sites', 'binding_sites', 'active_sites', 'pfam_domain', 'family']:
        fasta_data = {}
        for elem in data:
            if task in elem and elem[task] != None:
                fasta_data[elem['pid']] = elem['seq'][0]
        generate_fasta_file(fasta_data, save_path  + f"_{task}.fasta")



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
    parser.add_argument('--parse_test', action="store_true")
    args = parser.parse_args()

    #if not args.split_from_scratch:
    #    assert args.val_path != None

    paths_train = yaml.safe_load(open(args.paths_train, 'r'))
    paths_test = yaml.safe_load(open(args.paths_test, 'r'))

    """ CONFIGURING RUN  """
    #go_graph = get_go_graph(paths_train, return_data=False)
    go_graph=None

    ##""" GETTING GLOBAL PID TABLE FROM UNIPROT  """
    print(paths_train["frames"]+"/id_table.json.gz")
    assert os.path.exists(paths_train["frames"]+"/id_table.json.gz"), "swissprot from 202302 not processed!"
    pid_table_train, pid_table_test, uniref50_pid2cluster, uniref50_cluster2pid = load_files(paths_train, paths_test, load_uniref=False)

    #if args.path_to_test_cluster_pids != None:
    #    all_test_pids = set()
    #    with open(args.path_to_test_cluster_pids) as f:
    #        for line in f:
    #            all_test_pids.add(line[:-1])
    #    if args.additional_test_path != None:
    #        with open(args.additional_test_path) as f:
    #            for line in f:
    #                all_test_pids.add(line[:-1])
    #else:
    #    test_pids = get_temporal_test_pids(pid_table_train, pid_table_test, uniref50_pid2cluster)
    #    all_test_pids = expand_test_seqs(test_pids, uniref50_cluster2pid, uniref50_pid2cluster)

    #print("NUM TOTAL TEST PIDS + BALL (these will be removed from training):: {}".format(len(all_test_pids)))

    #if args.split_from_scratch:
    #    test_pid_centers = set()
    #    for pid in all_test_pids:
    #        test_pid_centers.add(uniref50_pid2cluster[pid])
    #    for path in paths:
    #        if "mmseq2-split10" in path: cluster_path = paths[path]


    #    val_pids, train_pids = get_validation_ids(cluster_path,val_percent=0.10,uniref50_cluster2pids=uniref50_cluster2pid, pid_table=pid_table_train) 
    #else:
    #    df = pd.read_csv(args.val_path)
    #    print(df)
    #    val_pids = df['pid'].to_numpy()
    #    print("num val_pids before filtering: {}".format(len(val_pids)))
    #    val_pids = [p for p in val_pids if p not in all_test_pids]
    #    print("num val_pids after filtering: {}".format(len(val_pids)))
    #    print()

    #    #train_pids = pd.read_csv(args.train_path)['pid'].to_numpy()
    #    val_pid_set = set(val_pids)
    #    train_pids = [pid for pid in pid_table_train if pid not in val_pid_set]
    #    print("num train_pids before filtering: {}".format(len(train_pids)))
    #    train_pids = [p for p in train_pids if p not in all_test_pids]
    #    #print("num train_pids after filtering: {}".format(len(train_pids)))
    #    #train_pids = [p for p in train_pids if p not in val_pid_set]


    #    val_and_cluster = set()
    #    val_not_in_uniref = set()
    #    for pid in val_pids:
    #        val_and_cluster.add(pid)
    #        if pid in uniref50_pid2cluster:
    #            cluster_center = uniref50_pid2cluster[pid]
    #            #val_and_cluster.update(uniref50_cluster2pid[cluster_center])
    #            cluster_pids = uniref50_cluster2pid[cluster_center]
    #            for p_ in cluster_pids:
    #                if p_ in pid_table_train:
    #                    val_and_cluster.add(p_)
    #        else:
    #            val_not_in_uniref.add(pid)
    #    with open("val_not_in_uniref50.txt", 'w') as f:
    #        for pid in val_not_in_uniref:
    #            f.write(f"{pid}\n")

    #    val_pids = list(val_and_cluster)

    #    train_pids = [p for p in train_pids if p not in val_and_cluster]
    #    print("num train_pids after filtering: {}".format(len(train_pids)))
    #            
    #    train_and_cluster = set()
    #    train_not_in_uniref = set()
    #    for pid in train_pids:
    #        train_and_cluster.add(pid)
    #        if pid in uniref50_pid2cluster:
    #            cluster_center = uniref50_pid2cluster[pid]
    #            cluster_pids = uniref50_cluster2pid[cluster_center]
    #            for p_ in cluster_pids:
    #                if p_ in pid_table_train:
    #                    train_and_cluster.add(p_)
    #        else:
    #            train_not_in_uniref.add(pid)
    #    with open("train_not_in_uniref50.txt", 'w') as f:
    #        for pid in train_not_in_uniref:
    #            f.write(f"{pid}\n")
    #    train_pids = list(train_and_cluster)

    #    val_pids = [p for p in val_pids if p not in all_test_pids] 
    #    train_pids = [p for p in train_pids if p not in all_test_pids]

    #    #assert len(val_and_cluster & train_and_cluster) == 0, "there is overlap between train and val! {}".format(len(val_and_cluster & train_and_cluster))
    #    overlap = set(val_pids) & set(train_pids)
    #    print("overlap between trian and val:::", len(overlap))
    #    with open("overlap.txt", 'w') as f:
    #        for pid in overlap:
    #            f.write(f"{pid}\n")


    #with open("val_set_mmseq10_uniref50.txt", 'w') as f:
    #    for pid in val_pids:
    #        f.write(pid + '\n')

    #with open("train_set_mmseq10_uniref50.txt", 'w') as f:
    #    for pid in train_pids:
    #        f.write(pid + '\n')
    #peer_pids = []
    #with open("/ssd005/projects/uniprot_aspuru/peer_pid_remove.txt") as f:
    #    for line in f:
    #        peer_pids.append(line[:-1])
    #peer_pids = set(peer_pids)
    if not args.parse_test:
        val_pids = []
        with open("/ssd005/projects/uniprot_aspuru/pids/val_set_mmseq10_uniref50.txt") as f:
            for line in f:
                val_pids.append(line[:-1])

        train_pids = []
        with open("/ssd005/projects/uniprot_aspuru/pids/train_set_mmseq10_uniref50.txt") as f:
            for line in f:
                train_pids.append(line[:-1])

    if args.parse_test:
        train_pids = []
        with open("/ssd005/projects/uniprot_aspuru/pids/sp_trainval_202302_202401testremoved.txt") as f:
            for line in f:
                train_pids.append(line[:-1])

        #train_pids = val_pids + train_pids

        print("num train pids: {}".format(len(train_pids)))
        #train_pids_peer = [p for p in train_pids if p in peer_pids]
        #print("peer_pids", len(train_pids_peer))
        #print("num train pids after removing peer: {}".format(len(train_pids)))

        test_pids = []
        missing = set()
        curr_and_dep_test = []
        #with open("/ssd005/projects/uniprot_aspuru/pids/sp_test_202311_test.txt") as f:
        with open("/ssd005/projects/uniprot_aspuru/pids/sp_test_202401_test.txt") as f:
            for line in f:
                pid = line[:-1]
                if pid not in pid_table_test:
                    missing.add(pid)
                else:
                    test_pids.append(pid)
                curr_and_dep_test.append(pid)
        print("missing test pids: ", len(missing)) # depracated 
        curr_and_dep_test = set(curr_and_dep_test)
        train_pids = [p for p in train_pids if p not in curr_and_dep_test]
        print(f"total num train after filtering:: {len(train_pids)}")

    if args.parse_test:
        save_csv(test_pids, "/ssd005/projects/uniprot_aspuru/datasets_alllen/test_set_sp202401", paths_test, pid_table_test, go_graph=go_graph)
        #save_csv(train_pids, "/ssd005/projects/uniprot_aspuru/datasets_alllen/trainval_set_mmseq10_uniref50_newtasks", paths_train, pid_table_train, go_graph=go_graph)
    else:
        save_csv(val_pids, "/ssd005/projects/uniprot_aspuru/datasets_alllen/val_set_mmseq10_uniref50_newtasks", paths_train, pid_table_train, go_graph=go_graph)
        save_csv(train_pids, "/ssd005/projects/uniprot_aspuru/datasets_alllen/train_set_mmseq10_uniref50_newtasks", paths_train, pid_table_train, go_graph=go_graph)

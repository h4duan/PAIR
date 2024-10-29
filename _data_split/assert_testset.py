import pandas as pd
import pickle
from get_test_pids import load_files, get_temporal_test_pids
import yaml
import argparse
import sys, os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils import expand_test_seqs

path_to_val = "/ssd005/projects/uniprot_aspuru/validation_sets/val_set_mmseq10_10per.csv"
val_df = pd.read_csv(path_to_val)
path_to_train = "/ssd005/projects/uniprot_aspuru/validation_sets/train_set_mmseq10_90per.csv"
train_df = pd.read_csv(path_to_train)

path_to_test = ""
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--parse_id", action="store_true", help="Run the id parser")

   """ CONFIGURING RUN  """
   args = parser.parse_args()
   paths_train = yaml.safe_load(open("../_config/paths.yml", 'r'))
   paths_test = yaml.safe_load(open("../_config/paths_test.yml", 'r'))

   """ GETTING GLOBAL PID TABLE FROM UNIPROT  """
   assert os.path.exists(paths_train["frames"]+"/id_table.json.gz"), "swissprot from 202302 not processed!"
   with open("/ssd005/projects/uniprot_aspuru/trembl_pid2file_map.pkl", 'rb') as f:
       trembl_pid2file = pickle.load(f)
   pid_table_train, pid_table_test, uniref50_pid2cluster, uniref50_cluster2pid = load_files(paths_train, paths_test)
   print("**************************")
   print("number of pids in train::", len(pid_table_train))
   print("val df", len(val_df))
   print("train df", len(train_df))
   print("**************************")
   test_pids = get_temporal_test_pids(pid_table_train, pid_table_test, uniref50_pid2cluster)
   pids_that_should_not_be_there = expand_test_seqs(test_pids, uniref50_cluster2pid, uniref50_pid2cluster)
   pids_in_trembl = 0
   for pid in pids_that_should_not_be_there:
       if pid in trembl_pid2file:
           pids_in_trembl += 1
   print("---------------------------")
   print("pids in trembl:::::", pids_in_trembl)
   exit()
   print("test pids", list(pids_that_should_not_be_there)[:10])

   val_pids_that_should_not_be_there = val_df[val_df['pid'].isin(pids_that_should_not_be_there)]
   assert len(val_pids_that_should_not_be_there) == 0

   train_pids_that_should_not_be_there = train_df[train_df['pid'].isin(pids_that_should_not_be_there)]
   assert len(train_pids_that_should_not_be_there) == 0

   val_pids = val_df['pid'].to_numpy()
   for pid in val_pids:
       assert pid not in pids_that_should_not_be_there

   train_pids = train_df['pid'].to_numpy()
   for pid in train_pids:
       assert pid not in pids_that_should_not_be_there
   print("check done -- no test pids in train")

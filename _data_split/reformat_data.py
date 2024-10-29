import argparse
import pandas as pd
import sys, os

pwd = os.path.dirname(os.getcwd())
sys.path.insert(0, pwd)

from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--label", choices=['EC', 'GO', 'names'], required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.path)
    print(df.columns)
    print(df['seq'])
    df['seq'] = df['seq'].apply(lambda x: reformat_seq(x))
    print(df['seq'])
    print(df)
    df[args.label] = df[args.label].apply(lambda x: reformat_label_list(x))
    df = df[df[args.label].notna()]
    print(df) 
    data = []
    total_seq_len = 0
    for ii, row in df.iterrows():
        if args.label == "names":
            row[args.label] = [name for name in row[args.label] if name != ""]
            if len(row[args.label]) == 0: continue
        total_seq_len += len(row['seq'])
        data.append({'input': row['seq'], 'target': row[args.label], 'split': 'val', 'pid': row['pid']})
    print("task:::", args.label)
    print("total sequence length:::", total_seq_len)
    print("num samples:::", len(data))

    save_dir = '/ssd005/projects/uniprot_aspuru/validation_sets/jsons/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = f"{args.label}_val.json"
    with open(os.path.join(save_dir, save_path), 'w') as f:
        json.dump(data, f)

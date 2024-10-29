import sys
import pickle
from tqdm import tqdm
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random
from tqdm import tqdm

def reformat_dict(x):
    return {'go_label': x[0], 'qualifier': x[1], 'evidence_tag': x[2], 'ancestors': x[3]}

def parser(paths, pid_table, cid_table):
    data = {}
    file_path, fact_date = get_file_path_and_date_from_key( "quickgo", paths )
    go_graph = get_go_graph(paths, return_data=False)
    source = "quickgo"
    fin = open(file_path)
    pid2fid = {}
    for line in tqdm(fin):
        if line.startswith("UniProtKB"):
            entry = line.split('\t')
            pid = entry[1]
            go_label = entry[4]
            qualifier = entry[3]
            tag = entry[6]
            #if tag != "IEA":
            if pid not in pid2fid:
                fid = random.getrandbits(128)
                pid2fid[pid] = fid
                data[ fid ] = { "fact_type": "protein_go" , "subjects": ["pid@"+pid], "content": set() , "date": fact_date, "source": source }
            ancestors = tuple(sorted(get_go_ancestors(go_graph, go_label)))
            data[pid2fid[pid]]["content"].add((go_label, qualifier, tag, ancestors))
    for fid in data:
        data[fid]['content'] = reformat_dict(data[fid]['content'])
    return data

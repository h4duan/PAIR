import random
import sys
sys.path.append('..')
from utils import *

def get_batch( frame, pid_table, cid_table, size, id_filter, args, parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]

    batch = []
    fids = list(frame.keys())
    """
    if parallel == "dp":
        random.shuffle(fids)
    else:
        if chunk_size * (rank + 2) > len(fids):
            fids = fids[chunk_size*rank:]
        else:
            fids = fids[chunk_size*rank:chunk_size*(rank+1)]
    """
    cluster = []

    #random.shuffle(fids)

    for fid in fids:

        pid = frame[fid]["subjects"][0][4:] #We know we only have 1 pid <=> function
        if pid not in pid_table:
            continue
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms)
        names = [ pid_table[pid]["name"] ]
        if len(seq) > args["protein_max_length"]:
            continue
        #print(frame[fid])
        text = frame[fid]["content"]["text"]
        #print(text)
        #print("---------")
        # ND: This is probably not going to be used for family(?)
        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."

        if pid not in id_filter["pids"]: batch.append( { "pid": pid, "fact_type": "protein_family" , "protein@1": seq, "text@1": text  })
        #if len(batch) == size: return batch
    #print(len(batch))
    #exit()
    return batch, cluster

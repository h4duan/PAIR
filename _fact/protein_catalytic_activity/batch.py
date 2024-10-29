import random
import sys
sys.path.append('..')
from utils import *

def get_batch( frame, fids, pid_table, cid_table, size, id_filter, args, parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    batch = []
    #fids = list(frame.keys())
    #random.shuffle(fids)
    for fid in fids:
        pid = frame[fid]["subjects"][0][4:] #We know we only have 1 pid <=> function
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms)
        if len(seq) > args["protein_max_length"]:
            continue
        reaction = random.choice(frame[fid]["content"]["reactions"])
        text = "A catalytic reaction of this protein is " + reaction + "."
        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."
        if pid not in id_filter["pids"]: 
            cluster += [pid_table[pid]["uniref50"]]
            batch.append( {"pid":pid, "fact_type": "protein_catalytic_activity" , "protein@1": seq, "text@1": text  })
        #if len(batch) == size: return batch, []
    return batch, []

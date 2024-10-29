import random
import sys
sys.path.append('..')
from utils import *

def get_batch( frame, fids, pid_table, cid_table, size, id_filter, args , parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    batch = []
 
    cluster = []

    #random.shuffle(fids)
    for fid in fids:
        pid = frame[fid]["subjects"][0][4:] #We know we only have 1 pid <=> function
        if pid not in pid_table:
            continue
        if "uniref50" not in pid_table[pid]:
            continue
        #cluster += [pid_table[pid]["uniref50"]]
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms)
        if len(seq) > args["protein_max_length"]:
            seq = random_crop_aaseq(seq, args["protein_max_length"])
        text  = []
        for cofactor in frame[fid]["content"]["cofactor_names"]:
            text += ["Cofactor: " + cofactor]
        if pid not in id_filter["pids"]: 
            for tt in text:
                cluster += [pid_table[pid]["uniref50"]]
                batch.append( { "fact_type": "protein_cofactor" , "protein@1": seq, "text@1": tt, "pid":pid  })
        #if len(batch) == size: return batch
    return batch, cluster

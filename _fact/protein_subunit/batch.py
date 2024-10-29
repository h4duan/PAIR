import random
import sys
sys.path.append('..')
from utils import *

def get_batch( frame, fids, pid_table, cid_table, size, id_filter, args , pid2cluster=None, parallel="dp", rank=0, chunk_size=0):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    batch = []
    #fids = list(frame.keys())
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
        if "uniref50" not in pid_table[pid]:
            continue
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms)
        if len(seq) > args["protein_max_length"]:
            seq = random_crop_aaseq(seq, args["protein_max_length"])
        #frame[fid]["content"]["subunits"]
        text = frame[fid]["content"]["subunits"]
        #if "(By similarity)" in text:
        #    continue
        new_text = []
        for tt in text:
            #if "(By similarity)" in text:
            #    continue
            new_text += tt.split(". ")
        if len(new_text) == 0:
            continue
        if pid not in id_filter["pids"]: 
            for tt in new_text:
                tt = remove_pubmed_substrings(tt)
                #if "(By similarity)" in tt:
                #    continue
                tt = remove_similarity_substrings(tt)
                #tt = "Subunit: " + tt
                cluster += [pid_table[pid]["uniref50"]]
                batch.append( { "fact_type": "protein_subunit" , "pid": pid, "protein@1": seq, "text@1": "Subunit: " + tt  })
        #if len(batch) == size: return batch
    return batch, cluster

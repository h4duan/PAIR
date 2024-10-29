import random
import sys
sys.path.append('..')
from utils import *
import random
def get_batch( frame, fids, pid_table, cid_table, size, id_filter, args, pid2cluster=None, return_all_seq=False, parallel="dp", rank=0, chunk_size=0):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    batch = []
    cluster = []
    for fid in fids:
        pid = frame[fid]["subjects"][0][4:] #We know we only have 1 pid <=> function
        if pid not in pid_table:
            continue
        if "uniref50" not in pid_table[pid]:
            continue
        #cluster += [pid_table[pid]["uniref50"]]
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms, return_all=return_all_seq)
        if return_all_seq:
            if len(seq[0]) > args["protein_max_length"]:
                seq = random_crop_aaseq(seq, args["protein_max_length"])
        else:
            if len(seq) > args["protein_max_length"]:
                seq = random_crop_aaseq(seq, args["protein_max_length"])
        #if args['joint']:
        text = "Alternative names: " + ", ".join(frame[fid]["content"]["alternative_names"])
        #else:
        #    text = "An alternative name of this protein is " + random.choice(frame[fid]["content"]["alternative_names"])
        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."
        if pid not in id_filter["pids"]: 
            cluster += [pid_table[pid]["uniref50"]]
            batch.append( { "fact_type": "protein_alternative_name" , "protein@1": seq, "text@1": text, "pid":pid  })
        #if len(batch) == size: return batch
    return batch, cluster

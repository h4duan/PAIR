import random
import sys
sys.path.append('..')
from utils import *
from transformers import AutoTokenizer

def get_batch(frame, fids, pid_table, cid_table, size, id_filter, args, parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    anonymize = args["anonymize"]
    protein_max_length = args["protein_max_length"]
    sample_one_sentence = args["sample_one_sentence"]
    batch = []
    cluster = []
    """
    fids = list(frame.keys())
    if parallel == "dp":
        random.shuffle(fids)
    else:
        if chunk_size * (rank + 2) > len(fids):
            fids = fids[chunk_size*rank:]
        else:
            fids = fids[chunk_size*rank:chunk_size*(rank+1)]
    cluster = []
    """
    for fid in fids:

        pid = frame[fid]["subjects"][0][4:] #We know we only have 1 pid <=> function
        if pid not in pid_table:
            continue
        if "uniref50" not in pid_table[pid]:
            continue
        #cluster += [pid_table[pid]["uniref50"]]
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms)
        if len(seq) > protein_max_length:
            seq = random_crop_aaseq(seq, protein_max_length)

        text = frame[fid]["content"]["text"]
        #if sample_one_sentence: 
        #    text = [np.random.choice(text)]
        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."

        if pid not in id_filter["pids"]: 
            for tt in text:
                if args['sample_function_only'] and "FUNCTION" not in tt['scopes']: continue
                new_tt = "Paper title: " + tt['title']
                cluster += [pid_table[pid]["uniref50"]]
                batch.append( { "pid": pid, "fact_type": "protein_papertitle" , "protein@1": seq, "text@1": new_tt  })
            
            #if len(batch) >= 10: 
            
            #print(batch)
            #    exit()
        #if len(batch) == size: return batch
    
    return batch, cluster

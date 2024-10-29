import random
import sys
sys.path.append('..')
from utils import *

def get_batch( frame, fids,pid_table, cid_table, size, id_filter, args, parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
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
        #keys = list(frame[fid]["content"].keys())
        #random.shuffle(keys)
        #key = keys[0]
        text = []
        for og in frame[fid]["content"].keys():
            if og != "scientific":
                continue
            og_name = remove_strain_substrings(frame[fid]["content"][og])
            #print(pid, og, og_name)
            text += ["Organism: " + og_name]
        #text = text[:-1].strip() + "."
        if pid not in id_filter["pids"]: 
            for tt in text:
                cluster += [pid_table[pid]["uniref50"]]
                batch.append( { "pid": pid, "fact_type": "protein_organism" , "protein@1": seq, "text@1": tt  })
        #if len(batch) == size: return batch
    #print(len(batch))
    #exit()
    return batch, cluster

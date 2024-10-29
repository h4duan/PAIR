import random
import sys
sys.path.append('..')
from utils import *

def get_batch( frame, fids, pid_table, cid_table, size, id_filter, args, name_processor=None, return_all_seq=False, parallel="dp", rank=0, chunk_size=0):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    batch = []
    #file1 = open("name_underscore.txt", "w")
    #fids = list(frame.keys())
    """
    if parallel == "dp":
        random.shuffle(fids)
    else:
        #print(fids[:5])
        if chunk_size * (rank + 2) > len(fids):
            fids = fids[chunk_size*rank:]
        else:
            fids = fids[chunk_size*rank:chunk_size*(rank+1)]
    """
    #cluster = []
    #print(len(fids))
    #random.shuffle(fids)
    cluster = []
    for fid in fids:
        pid = frame[fid]["subjects"][0][4:] #We know we only have 1 pid <=> function
        if pid not in pid_table:
            continue
        if "uniref50" not in pid_table[pid]:
            continue
        #cluster += [pid_table[pid]["uniref50"]]
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms, return_all=return_all_seq)
        #if return_all_seq:
        #    if len(seq[0]) > args["protein_max_length"]: continue
        #else:
        if len(seq) > args["protein_max_length"]:
            seq = random_crop_aaseq(seq, args["protein_max_length"])
        if args["preprocess"]:
            name = name_processor.preprocess_name(frame[fid]["content"]["recommended_name"])
        else:
            name = frame[fid]["content"]["recommended_name"]
        if len(name) == 0:
            continue
        text = "Recommended name: " + name + "."

        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."
        #if pid in id_filter["pids"]:
        #    print(f"{pid}\t{name}", file=file1)
        if pid not in id_filter["pids"]:
            #if args["upsample"]:
            #    cluster += [pid2cluster[pid]]
            cluster += [pid_table[pid]["uniref50"]]
            batch.append( { "fact_type": "protein_recommended_name" , "protein@1": seq, "text@1": text, "pid":pid  })
        #else:
        #    print("filtered in the validation set!")
        #if len(batch) >= size:
            #exit()
        #    return batch, cluster
    #exit()
    return batch, cluster

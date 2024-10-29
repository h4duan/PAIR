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
        #cluster += [pid_table[pid]["uniref50"]]
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms)
        if len(seq) > args["protein_max_length"]:
            seq = random_crop_aaseq(seq, args["protein_max_length"])
        binding_site_dict = frame[fid]["content"]
        #random.shuffle(binding_site_dict)
        text = []
        #if args['joint']:
            #text = "Binding sites:"
        for key in binding_site_dict:
            text += [f"Binding sites: {key.lower()}"]
            #text = text[:-1] + "."
        #else:
        
        #random_binding_site = random.choice(list(binding_site_dict.keys()))
        #    text= f"This protein has {len(binding_site_dict[random_binding_site])} {random_binding_site.lower()} binding site(s)."
        names = [ pid_table[pid]["name"] ]

        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."
        if pid not in id_filter["pids"]: 
            for tt in text:
                cluster += [pid_table[pid]["uniref50"]]
                batch.append( { "fact_type": "protein_binding_sites" , "protein@1": seq, "text@1": tt, 'pid':pid  })
        #if len(batch) == size: 
        #    return batch
    return batch, cluster

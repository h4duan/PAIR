import random
import sys
sys.path.append('..')
from utils import *
from transformers import AutoTokenizer

def get_batch( frame, fids, pid_table, cid_table, size, id_filter, args, parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    #anonymize = args["anonymize"]
    protein_max_length = args["protein_max_length"]
    #sample_one_sentence = args["sample_one_sentence"]
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
        if len(seq) > protein_max_length:
            seq = random_crop_aaseq(seq, protein_max_length)
        names = [ pid_table[pid]["name"] ]
        if pid_table[pid]["alternative_names"]: names += pid_table[pid]["alternative_names"]
        
        if "locations" in frame[fid]["content"]:
            #print(frame[fid]["content"]['locations'])
            text = []
            for tt in frame[fid]["content"]['locations']:
                text += [f"Subcellular locations: {tt}."]
            if use_organism and pid_table[pid]["organism"]:
                text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."
            if pid not in id_filter["pids"]: 
                for tt in text:
                    cluster += [pid_table[pid]["uniref50"]]
                    batch.append( { "pid": pid, "fact_type": "protein_subcellular_location" , "protein@1": seq, "text@1": tt  })
        """
        if "topologies" in frame[fid]["content"]:
            text = "Subcellular location topology: " + ", ".join(frame[fid]["content"]['topologies']) + "."
            if use_organism and pid_table[pid]["organism"]:
                text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."
            if pid not in id_filter["pids"]: 
                batch.append( { "pid": pid, "fact_type": "protein_subcellular_location" , "protein@1": seq, "text@1": text  })
        
        if "text" in frame[fid]["content"]:
            text = frame[fid]["content"]["text"][0]
            text = text.split(". ")
            #print(text)
            text = [remove_pubmed_substrings(tt) for tt in text]
            #text = [remove_similarity_substrings(tt) for tt in text]
            #if sample_one_sentence:
            #text = text.split(". ")

            text = [tt.strip() for tt in text] 
            if use_organism and pid_table[pid]["organism"]:
                text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."

            if pid not in id_filter["pids"]: 
                for tt in text:
                    if "(By similarity)" in text:
                        continue
                    cluster += [pid_table[pid]["uniref50"]]
                    batch.append( { "pid": pid, "fact_type": "protein_subcellular_location" , "protein@1": seq, "text@1": "Subcellular location text: " + tt  })
        """
    return batch, cluster

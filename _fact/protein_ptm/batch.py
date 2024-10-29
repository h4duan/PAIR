import random
import sys
sys.path.append('..')
from utils import *
from transformers import AutoTokenizer

def get_batch( frame, fids, pid_table, cid_table, size, id_filter, args, parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    anonymize = args["anonymize"]
    protein_max_length = args["protein_max_length"]
    sample_one_sentence = args["sample_one_sentence"]
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
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
            #print(len(seq))
            #continue
        names = [ pid_table[pid]["name"] ]
        if pid_table[pid]["alternative_names"]: names += pid_table[pid]["alternative_names"]

        text = frame[fid]["content"]["text"]
        #print(text)
        text = anonymize_prompt(text, names)
        text = remove_pubmed_substrings(text)
        #if "(By similarity)" in text:
        #    continue
        text = remove_similarity_substrings(text)
        #if sample_one_sentence:
            #print(text)
            #print(text.split(". "))
            #token_length = len(tokenizer(text, return_tensors='pt').input_ids[0])
            #print(token_length)
            #if token_length > 128:
        #if "(By similarity)" in text:
        #    continue
        text = [text.split(". ")[0]]
            #else:
            #text = [text]
        #if anonymize:
        #    text = anonymize_prompt( text, names  )

        text = [tt.strip() for tt in text] 
        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."

        #print(text, token_length)
        if pid not in id_filter["pids"]: 
            for tt in text:
                cluster += [pid_table[pid]["uniref50"]]
                new_tt = "PTM: " + tt
                #if "(By similarity)" in text:
                #    continue
                batch.append( { "pid": pid, "fact_type": "protein_ptm" , "protein@1": seq, "text@1": new_tt  })
        #if len(batch) == size: return batch
    
    return batch, cluster

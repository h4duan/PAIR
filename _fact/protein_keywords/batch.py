import random
import sys
sys.path.append('..')
from utils import *
from transformers import AutoTokenizer

def process_text_to_dict(paths):
    file_path,_ = get_file_path_and_date_from_key("kw-mapping", paths)
    with open(file_path) as f:
        text = f.read()
    entries = text.strip().split('//')
    kw_dict = {}

    for entry in entries:
        lines = entry.strip().split('\n')
        ac, id_, ca = None, None, None
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                code, content = parts
                if code == 'AC':
                    ac = content
                elif code == 'ID':
                    id_ = content
                elif code == 'CA':
                    ca = content
        
        kw_dict[ac] = {'category': ca, 'term': id_}

    return kw_dict

def get_batch(frame, fids, pid_table, cid_table, size, id_filter, args, parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    anonymize = args["anonymize"]
    protein_max_length = args["protein_max_length"]
    sample_one_sentence = args["sample_one_sentence"]

    paths = args["paths"]
    kw_dict = process_text_to_dict(paths)
    batch = []
    cluster = []
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

        text = frame[fid]["content"]['keywords']
        text = [t for t in text if kw_dict[t['@id']]['category'] not in ['Technical term.', 'Developmental stage.']]
        if not args['keep_go_kw']:
            text = [t for t in text if kw_dict[t['@id']]['category'] not in ['Biological process.', 'Cellular component.', 'Molecular function.']]
        #if sample_one_sentence:
        #    text=[np.random.choice(text)]

        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."

        #print(text, token_length)
        if pid not in id_filter["pids"]: 
            for tt in text:
                new_tt = "Keyword: " + tt["#text"]
                cluster += [pid_table[pid]["uniref50"]]
                batch.append( { "pid": pid, "fact_type": "protein_keywords" , "protein@1": seq, "text@1": new_tt  })
        #if len(batch) == size: return batch
    
    return batch, cluster

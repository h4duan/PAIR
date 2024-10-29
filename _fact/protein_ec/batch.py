import random
import sys
import copy
import numpy as np
np.random.seed(42)
sys.path.append('..')
from utils import *
from random import shuffle

def get_batch( frame, pid_table, cid_table, size, id_filter, args, return_all_seq=False, parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    protein_max_length = args["protein_max_length"]
    batch = []
    fids = list(frame.keys())
    #file1 = open('clean_train_1204.txt', 'w')
    if parallel == "dp":
        random.shuffle(fids)
    else:
        if chunk_size * (rank + 2) > len(fids):
            fids = fids[chunk_size*rank:]
        else:
            fids = fids[chunk_size*rank:chunk_size*(rank+1)]
    cluster = []
    jj = 0
    for fid in fids:
        pid = frame[fid]["subjects"][0][4:] #We know we only have 1 pid <=> function
        if pid not in pid_table:
            #jj += 1
            #print(f"{pid} not in uniref, {jj}")
            #exit()
            continue
        seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms, return_all=return_all_seq)
        if len(seq) > args["protein_max_length"]: 
            seq = random_crop_aaseq(seq, args["protein_max_length"])
        key = random.choices( [ "ec_numbers", "ec_texts"], weights=[ 1 - float(args["p"]), float(args["p"]) ], k =1  )[0]
        if key == "ec_numbers":
            #text = "This protein has the EC number: "
            prompt = "This protein has the EC number:"
            ec_numbers = []
            for ec in frame[fid]["content"][key]:
                #text = " ".join(random.choice( frame[fid]["content"][key]  ).split("."))
                if args['filter_n'] and "n" in ec:
                    continue
                if len(ec.split(".")) != 4:
                    continue
                else:
                    ec_numbers += [" ".join(ec.split("."))]
            #ec_numbers = set(ec_numbers)
            old_ec_numbers = ";".join(ec_numbers)
            shuffle(ec_numbers)
            if len(ec_numbers) == 0:
                continue
            #ec_numbers = set(ec_numbers)
            elif args["joint"]:
                text = prompt
                for ec in ec_numbers:
                    text += " " + ec + ","
                text = text[:-1] + "."
            else:
                text = [prompt + " " + ec + "." for ec in ec_numbers]
        elif not args["joint"] and key == "ec_texts":
            text = random.choice( frame[fid]["content"][key]  )
        else:
            text = ''
            for ec in frame[fid]["content"][key]:
                 text += random.choice( frame[fid]["content"][key]  ) + ' '
            text = text[:-1]
        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."
        if pid not in id_filter["pids"]:
            if type(text) is list:
                for tt in text:
                    try:
                        if args["upsample"]:
                            cluster += [tt]
                            #cluster += [text[len(prompt):]]
                        batch.append( { "fact_type": "protein_ec" , "protein@1": seq, "text@1": tt, "pid": pid  })
                    #print(pid2cluster[pid])
                        #cluster += [pid2cluster[pid]]
                    except:
                        print(f"pid {pid} not in uniref cluster ")
                        continue
            elif type(text) is str:
                #print(pid, text)
                #print(text)
                batch.append( { "fact_type": "protein_ec" , "protein@1": seq, "text@1": text, "pid": pid  })
                if args["upsample"]:
                    if pid2cluster is None:
                        cluster += [old_ec_numbers]
                    else:
                        cluster += [pid2cluster[pid]]
    #exit()
    return batch, cluster

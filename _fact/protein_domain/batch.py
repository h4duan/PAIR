import random
import sys
sys.path.append('..')
from utils import *

def get_pfam2name(paths):
    file_path,_ = get_file_path_and_date_from_key("pfam-mapping", paths)
    df = pd.read_csv(file_path, sep='\t', header=None)
    pfam2name = dict(zip(df[3], df[4]))
    return pfam2name

def get_interpro2name(paths):
    file_path,_ = get_file_path_and_date_from_key("interpro-mapping", paths)
    df = pd.read_csv(file_path, sep='\t')
    interpro2name = dict(zip(df.ENTRY_AC, df.ENTRY_NAME))
    return interpro2name


def get_batch( frame, fids, pid_table, cid_table, size, id_filter, args , parallel="dp", rank=0, chunk_size=0, pid2cluster=None):
    paths = args["paths"]
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    pfam2name = get_pfam2name(paths)
    interpro2name = get_interpro2name(paths)

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
        #names = [ pid_table[pid]["name"] ]
        #text = frame[fid]["content"]["text"]
        #print(frame[fid])
        prompts = []
        #prompt = "Domains: "
        #prompt_saved = "This protein contains "
        if args['parse_interpro'] and 'interpro_ids' in frame[fid]["content"]:
            for tt in frame[fid]["content"]['interpro_ids']:
                if tt in interpro2name:
                    prompts += ['Domain: ' + interpro2name[tt]]
        if args['parse_pfam'] and 'pfam_domains' in frame[fid]["content"]:
            for tt in frame[fid]["content"]['pfam_domains']:
                assert len(tt) == 1, tt
                id_ = list(tt.keys())[0]
                if id_ in pfam2name:
                    prompts += ['Domain: ' + pfam2name[id_]]
                else:
                    prompts += ['Domain: ' + id_]
        if 'domain_text' in frame[fid]["content"]:
            text = frame[fid]["content"]["domain_text"]
            text = remove_pubmed_substrings(text)
            if "(By similarity)" in text:
                continue 
            #text = remove_similarity_substrings(text)
            #text = anonymize
            text = text.split(". ")
            text = [tt.strip() for tt in text]
            for tt in text:
                prompts += ['Domain: ' + tt]
            #for tt in frame[fid]["content"]["domain"]:

            #    domain = tt[len(prompt_saved)+2:-len(" domain(s) ")]
            #    domain = " ".join(domain.split("_"))
            #    text += ["Domain: " + domain]
        #if not args['joint']:
        #    prompts = [random.choice(prompts)]
        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."

        if pid not in id_filter["pids"]: 
            for tt in prompts:
                cluster += [pid_table[pid]["uniref50"]]
                batch.append( {"pid":pid, "fact_type": "protein_domain" , "protein@1": seq, "text@1": tt  })
        #if len(batch) == size: return batch, cluster

    return batch, cluster

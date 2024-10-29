import random
import ast
import sys
import numpy as np
sys.path.append('..')
from utils import *
import obonet
import urllib.request, json

EXP_TAGS = set(["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC", "HTP", "HDA", "HMP", "HGI", "HEP"])

def parse_ecoids(paths):
    eco2tag = {}
    file_path,_ = get_file_path_and_date_from_key("eco-mapping", paths)
    with open(file_path) as f:
        for line in f:
            if line[0] == "#": continue
            elems = line[:-1].split('\t')
            eco_id = elems[0]
            tag = elems[1]
            eco2tag[eco_id] = tag
    return eco2tag


def modifier_and_qualifier(qualifier):
    #file1 = open('go_train.txt', 'w')
    modifier = "" # check for all qualifiers
    if (" in" in qualifier or "part of" in qualifier) and "is " not in qualifier:
        modifier = " is"
    if "NOT" in qualifier:
        qualifier = qualifier.split("|")[1]
        if qualifier[:3] == "is ":
            modifier =" is not"
            qualifier = qualifier[3:]
        elif (" in" in qualifier or "part of" in qualifier):
            modifier =" is not"
        else:
            modifier=" does not"
            qualifier = qualifier.split()
            if len(qualifier) == 1:
                qualifier = qualifier[0][:-1]
            else:
                qualifier = qualifier[0][:-1] + " " + ' '.join(qualifier[1:])
    return modifier, qualifier

def go2text( go, go_graph ):
    tag2text = {
        "EXP": "experiment",
        "IDA": "direct assay",
        "IPI": "physical interaction",
        "IMP": "mutant phenotype",
        "IGI": "genetic interaction",
        "IEP": "expression pattern",
        "IBA": "biological aspect of ascendent",
        "IBD": "biological aspect of descendent",
        "IKR": "key residues",
        "IRD": "rapid divergence",
        "ISS": "sequence or structural similarity",
        "ISO": "sequence orthology",
        "ISA": "sequence alignment",
        "ISM": "sequence model",
        "IGC": "genomic context",
        "RCA": "reviewed computational analysis",
        "TAS": "traceable author statement",
        "NAS": "non-traceable author statement",
        "IC": "curator",
        "ND": "no biological data",
        "IEA": "electronic annotation",
        "HTP": "high throughput experiment",
        "HDA": "high throughput direct assay",
        "HMP": "high throughput mutant phenotype",
        "HGI": "high throughput genetic interaction",
        "HEP": "high throughput expression pattern"
    }
    #print(go_graph)
    try:
        node = go_graph[go["go_label"]]
    except:
#        print("here")
        print(go["go_label"])
        return ""
    #print(node)
    text = node["name"]
    return text
    #qualifier = ' '.join(go["qualifier"].split('_'))
    #modifier, qualifier = modifier_and_qualifier(qualifier)
    #evidence = tag2text[go["evidence_tag"]]
    #return f"This protein{modifier} {qualifier} {text} (annotation inferred from {evidence})."

def filter_go_label(go_label):
    if go_label[:3] == "GO:":
        return go_label[3:]

def get_ancestor_path(go_label, go_graph, max_hop=-1, use_text=False):
    ## hop refers to how many ancestors to get. if hop is -1 then it grabs infinite path
    original_go_label = go_label
    anno = go_label
    if use_text:
        anno = go_graph[go_label]['name']
    path = "{}".format(anno)
    hop = 0
    while True: 
        if hop == max_hop: # condition not met by default if max_hop is -1 (infinite)
            break
        parents = []
        if 'is_a' in go_graph[go_label]:
            for parent in go_graph[go_label]['is_a']:
                parents.append((parent, 'is_a'))
        if 'relationship' in go_graph[go_label]:
            for parent in go_graph[go_label]['relationship']:
                relationship, id_ = parent.split()
                parents.append((id_, relationship))
        parent_idx = np.random.choice(len(parents))
        parent = parents[parent_idx]
        anno = parent[0]
        if use_text:
            anno = go_graph[anno]['name']
        path += ", which {} {}".format(" ".join(parent[1].split("_")), anno)
        hop += 1
        go_label = parent[0]
        if go_label in ["GO:0003674", "GO:0008150", "GO:0005575"]:
            break
        if hop == 100:
            print("path not terminating..........", original_go_label)
            break
    return path 

def get_batch( frame, pid_table, cid_table, size, id_filter, args, return_all_seq=False, parallel="dp", rank=0, chunk_size=0,pid2cluster=None):
    augment_with_variants = args["augment_with_variants"]
    augment_with_isoforms = args["augment_with_isoforms"]
    use_organism = args["use_organism"]
    paths = args["paths"]
    #go_graph = get_go_graph(paths)
    go_graph, go_path = get_go_graph(paths, return_data=False, return_path=True)
    #go_rels = Ontology('data/go.obo', with_rels=True)
    go_rels= Ontology(go_path, with_rels=True)
    eco2gotag = parse_ecoids(paths)
    batch = []
    cluster = []
    fids = list(frame.keys())
    random.shuffle(fids)
    for fid in fids:
        pid = frame[fid]["subjects"][0][4:] #We know we only have 1 pid <=> function
        if pid in id_filter['pids']: continue

        if pid not in pid_table:
            continue
            with urllib.request.urlopen( "https://rest.uniprot.org/uniprotkb/" + pid + ".json"  ) as url:
                entry = json.load(url)
                seq = entry["sequence"]["value"]
        else:
            seq = generate_aaseq(pid_table[pid], augment_with_variants, augment_with_isoforms, return_all=return_all_seq)

        #if return_all_seq:
        #    if len(seq[0]) > args["protein_max_length"]: continue
        #else:
        if len(seq) > args["protein_max_length"]: 
            seq = random_crop_aaseq(seq, args["protein_max_length"])

        use_text = np.random.choice(2, p=[1-float(args["p"]),float(args["p"])])

        #if pid2cluster is not None and pid not in pid2cluster:
        #    continue

        if args["joint"]:
            if use_text:
                #print("here")
                text = "This protein has the following GO annotations: "
                len_p = len(text)
                go_anno = set()
                for go in frame[fid]["content"]:
                    go_id, eco_id, go_text = go
                    go_tag = eco2gotag[eco_id]
                    if args['filter_exp_tags'] and go_tag not in EXP_TAGS: continue
                    go_anno.add(go_text)
                    #if go["go_label"] in go_graph: #and go_graph[go["go_label"]]["namespace"] == args["subontology"]:
                    #    go_anno.add(go2text(go, go_graph))
                
                #if args['filter_exp_tags'] and len(text) == len_p:
                #    continue # pid has no experimental tags, don't add pid
                if args['incl_ancestors']:
                    print("NOT IMPLEMENTED!!!!! exiting now........")
                    exit()
                    for go in frame[fid]["content"]:
                        go_id, eco_id, go_text = go
                        go_tag = eco2gotag[eco_id]
                        if args['filter_exp_tags'] and  go_tag not in EXP_TAGS: continue
                        #for ancestor in go['ancestors']:
                        #    if ancestor in go_graph: # and go_graph[ancestor]["namespace"] == args["subontology"]:
                        #        go_anno.add(go_graph[ancestor]["name"])
            else:
                text = "This protein has the following GO annotations: "
                go_anno = set()
                len_p = len(text)
                for go_id in frame[fid]["content"]:
                    eco_id = frame[fid]["content"][go_id]['evidence']
                    go_text = frame[fid]["content"][go_id]['term']
                    go_tag = eco2gotag[eco_id]

                    if args['filter_exp_tags'] and go_tag not in EXP_TAGS: continue
                    go_anno.add(filter_go_label(go_id))
                    if args['incl_ancestors']:
                        ancestors = go_rels.get_anchestors(go_id)
                        for ancestor in ancestors: 
                            go_anno.add(filter_go_label(ancestor))
                        #for ancestor in go['ancestors']:
                        #    if ancestor in go_graph:# and go_graph[ancestor]["namespace"] == args["subontology"]:
                        #        go_anno.add(filter_go_label(ancestor))                   
        else:
            print("NOT IMPLEMENTED!!!!! exiting now........")
            exit()
            if args['sample_path']:
                for go in frame[fid]["content"]:
                    if args['filter_exp_tags'] and go['evidence_tag'] not in EXP_TAGS: continue
                    if go['go_label'] not in go_graph: continue
                    max_hop = args['max_hop']
                    text = "This protein has the GO annotation: {}.".format(get_ancestor_path(go['go_label'], go_graph, max_hop, use_text))
            else:
                all_goa = [frame[fid]["content"]]
                if args['filter_exp_tags']:
                    all_goa = [goa["go_label"] for goa in all_goa if goa['evidence_tag'] in EXP_TAGS]
                    if len(all_goa) == 0: continue # there are no experimental go labels, don't add to batch
                for go in frame[fid]["content"]:
                    if args['filter_exp_tags'] and go['evidence_tag'] not in EXP_TAGS: continue
                    if args['incl_ancestors']:
                        for ancestor in go['ancestors']:
                            all_goa += [ancestor]
                all_goa = set(all_goa)
                go = random.choice(all_goa)
                if go not in go_graph:
                    print(go)
                    continue
                if use_text:
                    anno = go_graph[go]["name"]
                    text = f"This protein has the GO annotation: {anno}"
                else:
                    text = "This protein has the GO annotation {}.".format(go)
        if use_organism and pid_table[pid]["organism"]:
            text +=  " it is found in the organism of " + generate_organism_name( pid_table[pid]["organism"]  ) + "."
        if pid not in id_filter["pids"]: 
            #print(pid, text[len("This protein has the following GO annotations: "):], file=file1)
            #print(text)
            #print(text, file=file1)
            if args["joint"]:
                #print(go_anno)
                text = "This protein has the GO annotations:"
                if len(go_anno) == 0:
                    continue
                else:
                    go_anno = list(go_anno)
                    random.shuffle(go_anno)
                for goa in go_anno:
                    text += " \"" + goa + "\""
                #if pid2cluster is not None:
                #    cluster += [pid2cluster[pid]]
                batch.append( { "fact_type": "protein_go" , "protein@1": seq, "text@1": text + ".", "pid":pid  })
            else:
                batch.append( { "fact_type": "protein_go" , "protein@1": seq, "text@1": text, "pid":pid  })

        if len(batch) >= size: 
            return batch, cluster
    return batch, cluster



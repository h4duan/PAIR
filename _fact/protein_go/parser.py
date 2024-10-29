import sys
sys.path.append('..')
from utils import *
import utils
import gzip
import xmltodict
import random
from functools import partial

def get_protein_go_from_uniprot(paths, protein):
    #ec_num2text = utils.ec_num_to_text( paths  )
    go_label = {}
    df = protein['entry']
    #print("here")
    fact_attr = {}
    if "dbReference" in df and type(df["dbReference"]) is list:
        for d in df["dbReference"]:
            #print(d)
            assert type(d) == dict
            if d['@type'] == "GO":
                g_term = d['@id']
                go_label[g_term] = {}
                d_property = d["property"]
                for d_p in d_property:
                    if d_p["@type"] == "evidence":
                        go_label[g_term]["evidence"] = d_p["@value"]
                    if d_p["@type"] == "term":
                        go_label[g_term]["term"] = d_p["@value"]
                    #if len(g_term) > 0 and len(g_text) > 0 and len(g_evidence) > 0: 
                    #    go_label += [(g_term, g_evidence, g_text)]
                    #    continue
                #print(d)
    if len(go_label) > 0:
        fact_attr["content"] = go_label
        #print(fact_attr)
    #exit()
    return [fact_attr]         
    #return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    get_protein_go_from_uniprot_with_paths = partial(get_protein_go_from_uniprot, paths)
    data = parse_sprot_facts( "protein_go", file_path, file_date, source, get_protein_go_from_uniprot_with_paths, pid_table )
    return data

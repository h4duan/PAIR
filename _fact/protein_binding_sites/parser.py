import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_binding_sites_from_uniprot(protein):
    df = protein["entry"]
    binding_sites = {}
    if 'feature' in df:
        if type(df['feature']) != list:
            df['feature'] = [df['feature']]
        for feat in df['feature']:
            if feat["@type"] == "binding site":
                ligand=feat['ligand']['name']
                if ligand not in binding_sites:
                    binding_sites[ligand] = {}
                loc=None
                if "position" in feat['location'] and "@position" in feat['location']['position']:
                    loc = feat['location']['position']['@position']
                elif 'begin' in feat['location'] and "@position" in feat['location']['begin']:
                    start= feat['location']['begin']['@position']
                    if 'end' in feat['location'] and "@position" in feat['location']['end']:
                        loc = (start, feat['location']['end']['@position'])
                    else:
                        loc = start
                if 'label' in feat['ligand']:
                    label=feat['ligand']['label']
                else:
                    label='1'
                if label not in binding_sites[ligand]:
                     binding_sites[ligand][label] = []
                binding_sites[ligand][label].append(loc)
    content = {}
    if len(binding_sites) > 0: content = binding_sites
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_binding_sites", file_path, file_date, source, get_binding_sites_from_uniprot, pid_table)
    return data


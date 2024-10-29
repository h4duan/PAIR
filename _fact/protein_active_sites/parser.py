import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_active_sites_from_uniprot(protein):
    df = protein["entry"]
    active_sites = {}
    if 'feature' in df:
        if type(df['feature']) != list:
            df['feature'] = [df['feature']]
        for feat in df['feature']:
            if feat["@type"] == "active site":
                if '@description' not in feat: continue
                active_site=feat['@description']
                if active_site not in active_sites:
                    active_sites[active_site] = []

                if "position" in feat['location'] and "@position" in feat['location']['position']:
                    loc = feat['location']['position']['@position']
                elif 'begin' in feat['location'] and "@position" in feat['location']['begin']:
                    start= feat['location']['begin']['@position']
                    loc=None
                    if 'end' in feat['location'] and "@position" in feat['location']['end']:
                        loc = (start, feat['location']['end']['@position'])
                    else:
                        loc = start
                active_sites[active_site].append(loc)

    content = {}
    if len(active_sites) > 0: content = active_sites
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_active_sites", file_path, file_date, source, get_active_sites_from_uniprot, pid_table)
    return data


import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_sites_from_uniprot(protein):
    df = protein["entry"]
    sites = {}
    if 'feature' in df:
        if type(df['feature']) != list:
            df['feature'] = [df['feature']]
        for feat in df['feature']:
            if feat["@type"] == "site":
                if '@description' not in feat: continue
                site=feat['@description']
                if site not in sites:
                    sites[site] = []
                if "position" in feat['location'] and "@position" in feat['location']['position']:
                    loc = feat['location']['position']['@position']
                elif 'begin' in feat['location'] and "@position" in feat['location']['begin']:
                    start= feat['location']['begin']['@position']
                    loc=None
                    if 'end' in feat['location'] and "@position" in feat['location']['end']:
                        loc = (start, feat['location']['end']['@position'])
                    else:
                        loc = start
                sites[site].append(loc)

    content = {}
    if len(sites) > 0: content = sites
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_sites", file_path, file_date, source, get_sites_from_uniprot, pid_table)
    return data


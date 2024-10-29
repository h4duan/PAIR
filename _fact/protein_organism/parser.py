import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_organism_from_uniprot(protein):
    content = {}
    if type(protein["entry"]["organism"]["name"]) != list:
        protein["entry"]["organism"]["name"] = [protein["entry"]["organism"]["name"]]
    for org in protein["entry"]["organism"]["name"]:
        content[org["@type"]] = org["#text"]
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_organism", file_path, file_date, source, get_organism_from_uniprot, pid_table)
    return data

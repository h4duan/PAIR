import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_recommended_name_from_uniprot(protein):
    content = {}
    recommendedname = ""
    if "protein" in protein["entry"] and "recommendedName" in protein["entry"]["protein"]:
        if type(protein["entry"]["protein"]["recommendedName"]["fullName"]) == str:
            recommendedname = protein["entry"]["protein"]["recommendedName"]["fullName"]
        else:
            assert type(protein["entry"]["protein"]["recommendedName"]["fullName"]) == dict
            recommendedname = protein["entry"]["protein"]["recommendedName"]["fullName"]["#text"]
    assert type(recommendedname) == str
    if recommendedname != "": content["recommended_name"] = recommendedname
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_recommended_name", file_path, file_date, source, get_recommended_name_from_uniprot, pid_table)
    return data


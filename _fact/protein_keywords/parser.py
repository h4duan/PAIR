import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random


def get_keywords_from_uniprot(protein):
    content = {}
    if "keyword" in protein["entry"]:
        keywords = protein["entry"]["keyword"]
        if type(keywords) == dict: keywords = [keywords]
        assert type(keywords) == list
        if len(keywords) > 0:
            content = {"keywords": keywords}
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_keywords", file_path, file_date, source, get_keywords_from_uniprot, pid_table)
    return data

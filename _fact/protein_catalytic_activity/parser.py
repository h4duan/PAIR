import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_catalytic_activity_from_uniprot(protein):
    reactions = []
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "catalytic activity":
                assert type(comm["reaction"]["text"]) == str
                reactions += [comm["reaction"]["text"]]
    content = {}
    if len(reactions) > 0: content["reactions"] = reactions
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_catalytic_activity", file_path, file_date, source, get_catalytic_activity_from_uniprot, pid_table)
    return data

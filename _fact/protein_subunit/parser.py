import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_subunit_from_uniprot(protein):
    subunits = []
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "subunit":
                if type(comm["text"]) == str:
                    subunits += [ comm["text"] ]
                else:
                    assert type(comm["text"]["#text"]) == str
                    subunits += [ comm["text"]["#text"] ]
    content = {}
    if len(subunits) > 0: content["subunits"] = subunits
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_subunit", file_path, file_date, source, get_subunit_from_uniprot, pid_table )
    return data

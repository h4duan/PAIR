import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_alternative_names_from_uniprot(protein):
    names = []
    if "protein" in protein["entry"] and "alternativeName" in protein["entry"]["protein"]:
        alternativenames = protein["entry"]["protein"]["alternativeName"]
        if type(alternativenames) == dict:
            if type(alternativenames["fullName"]) is str:
                 names = [alternativenames["fullName"]]
            else:
                assert type(alternativenames["fullName"]) is dict
                names = [alternativenames["fullName"]["#text"]]
        else:
            assert type(alternativenames) == list
            for nn in alternativenames:
                if type(nn["fullName"]) is str:
                    names += [nn["fullName"]]
                else:
                    assert type(nn["fullName"]) is dict
                    names += [nn["fullName"]["#text"]]

    assert type(names) == list
    for name in names:
        assert type(name) == str
    content = {}
    if len(names) > 0: content["alternative_names"] = names
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_alternative_names", file_path, file_date, source, get_alternative_names_from_uniprot, pid_table)
    return data


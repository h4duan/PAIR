import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_family_from_uniprot(protein):
    content = {}
    family = ""
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "similarity":
                if type(comm["text"]) == str: comm["text"] = { "#text": comm["text"]  }
                family += comm["text"]["#text"] + " "
                if len(family) > 0:
                    if family[-1] == " ": family = family[:-1]
                if family != "":
                    content = {"text": family}
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_family", file_path, file_date, source, get_family_from_uniprot, pid_table)
    return data

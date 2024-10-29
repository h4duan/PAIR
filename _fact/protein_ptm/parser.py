import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_ptm_from_uniprot(protein):
    content = {}
    ptm = []
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "PTM":
                if type(comm["text"]) == str: comm["text"] = { "#text": comm["text"]  }
                #ptm += comm["text"]["#text"] + " "
                text = comm["text"]["#text"]
                if text[-1] != ".": 
                    text += "."
                ptm.append(text)
                #if len(ptm) > 0:
                #    if ptm[-1] == " ": ptm = ptm[:-1]
                #if ptm != "":
                #    content = {"text": ptm}
    if len(ptm) > 0:
        content['text'] = " ".join(ptm)
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_ptm", file_path, file_date, source, get_ptm_from_uniprot, pid_table)
    return data

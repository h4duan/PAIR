import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_tissue_specificity_from_uniprot(protein):
    content = {}
    tissue_specificity = []
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "tissue specificity":
                if type(comm["text"]) == str: comm["text"] = { "#text": comm["text"]  }
                text = comm["text"]["#text"]
                if text[-1] != ".":
                    text += "."
                tissue_specificity.append(text)
                #tissue_specificity += comm["text"]["#text"] + " "
                #if len(tissue_specificity) > 0:
                #    if tissue_specificity[-1] == " ": tissue_specificity = tissue_specificity[:-1]
                #if tissue_specificity != "":
                #    content = {"text": tissue_specificity}
    if len(tissue_specificity) > 0:
        content = {"text": ' '.join(tissue_specificity)}
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_tissue_specificity", file_path, file_date, source, get_tissue_specificity_from_uniprot, pid_table)
    return data

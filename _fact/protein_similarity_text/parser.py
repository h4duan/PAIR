import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_similarity_text_from_uniprot(protein):
    content = {}
    similarity_text = []
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "similarity":
                if type(comm["text"]) == str: comm["text"] = { "#text": comm["text"]  }
                #similarity_text += comm["text"]["#text"] + " "
                text = comm["text"]["#text"]
                if text[-1] != ".": 
                    text += "."
                similarity_text.append(text)
                #if len(similarity_text) > 0:
                #    if similarity_text[-1] == " ": similarity_text = similarity_text[:-1]
                #if similarity_text != "":
                #    content = {"text": similarity_text}
    if len(similarity_text) > 0:
        content = {"text": " ".join(similarity_text)}
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_similarity_text", file_path, file_date, source, get_similarity_text_from_uniprot, pid_table)
    return data

import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_activity_regulation_text_from_uniprot(protein):
    content = {}
    activity_regulation_text = []
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "activity regulation":
                if type(comm["text"]) == str: comm["text"] = { "#text": comm["text"]  }
                text = comm["text"]["#text"]
                if text[-1] != ".":
                    text += "."
                activity_regulation_text.append(text)
                #activity_regulation_text += comm["text"]["#text"] + " "
                #if len(activity_regulation_text) > 0:
                #    if activity_regulation_text[-1] == " ": activity_regulation_text = activity_regulation_text[:-1]
                #if activity_regulation_text != "":
                #    content = {"text": activity_regulation_text}
    if len(activity_regulation_text) > 0:
        content = {'text': ' '.join(activity_regulation_text)}
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_activity_regulation_text", file_path, file_date, source, get_activity_regulation_text_from_uniprot, pid_table)
    return data

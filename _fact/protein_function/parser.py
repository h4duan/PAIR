import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_function_from_uniprot(protein):
    content = {}
    function = []
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        function = []
        for comm in comment:
            if comm["@type"] == "function":
                if type(comm["text"]) == str: comm["text"] = { "#text": comm["text"]  }
                #function += comm["text"]["#text"] + " "
                text = comm["text"]["#text"]
                if text[-1] != ".":
                    text += "."
                function += [text]
                #if len(function) > 0:
                #    if function[-1] == " ": function = function[:-1]
                #if function != "":
                #    content = {"text": function}
    if len(function) > 0:
        content = {"text": function}
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_function", file_path, file_date, source, get_function_from_uniprot, pid_table)
    return data

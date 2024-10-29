import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_subcellular_location_from_uniprot(protein):
    content = {}
    function = ""
    locations = set()
    topologies = set()
    texts = set()
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "subcellular location":
                if 'subcellularLocation' in comm:
                    assert type(comm['subcellularLocation']) != str, comm
                    if type(comm['subcellularLocation']) == dict:
                        comm['subcellularLocation'] = [comm['subcellularLocation']]
                    for elem in comm['subcellularLocation']:
                        for category in ['location', 'topology']:
                            if category in elem:
                                loc = elem[category]
                                if type(loc) != list: loc= [loc]
                                for l in loc:
                                    assert type(l) in [dict, str], (l, elem)
                                    if type(l) == dict:
                                        fact = l['#text']
                                    elif type(l) == str:
                                        fact = l
                                    if category == "location":
                                        locations.add(fact)
                                    elif category == "topology":
                                        topologies.add(fact)
                if "text" in comm:
                    text_ = comm['text']
                    if type(text_) != list: text_ = [text_]
                    for t in text_:
                        assert type(t) in [dict, str], (t, elem)
                        if type(t) == dict:
                            fact = t['#text']
                        elif type(t) == str:
                            fact = t
                        texts.add(fact)



                if len(locations) > 0:
                    content['locations'] = list(locations)
                if len(topologies) > 0:
                    content['topologies'] = list(topologies)
                if len(texts) > 0:
                    content['text'] = list(texts)
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_subcellular_location", file_path, file_date, source, get_subcellular_location_from_uniprot, pid_table)
    return data

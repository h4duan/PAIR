import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random

def get_domain_from_uniprot(protein):
    pfam_domains=[]
    interpro_ids = {}
    domain_text = []
    if "comment" in protein["entry"]:
        comment = protein["entry"]["comment"]
        if type(comment) == dict: comment = [comment]
        for comm in comment:
            if comm["@type"] == "domain":
                if type(comm["text"]) == str: comm["text"] = { "#text": comm["text"]  }
                text= comm["text"]["#text"]
                if text[-1] != ".":
                    text += "."
                domain_text.append(text)
                #domain_text += comm["text"]["#text"] + " "
                #if len(domain_text) > 0:
                #    if domain_text[-1] == " ":
                #domain_text = domain_text[:-1]

    if "dbReference" in protein["entry"]:
        dbRef = protein["entry"]["dbReference"]
        if type(dbRef) == dict: dbRef = [dbRef]
        for ref in dbRef:
            if ref["@type"] == "Pfam":
                prop_ = ref["property"]
                assert type(prop_) == list, ref
                #dmn_txt = 'This protein contains ' + prop_[1]['@value'] + ' ' + prop_[0]['@value'] + ' domain(s).'
                pfam_domains.append({prop_[0]['@value']: prop_[1]['@value']})
            elif ref["@type"] == "InterPro":
                interpro_id = ref['@id']
                if 'property' in ref: assert type(ref['property']) == dict
                short_name = ref['property']['@value']

                if interpro_id in interpro_ids:
                    assert interpro_ids[interpro_id] == short_name
                interpro_ids[interpro_id] = short_name
    content = {}
    if len(domain_text) > 0:
        content['domain_text'] = ' '.join(domain_text)
    if len(pfam_domains) > 0:
        content["pfam_domains"] = pfam_domains
    if len(interpro_ids) > 0:
        content['interpro_ids'] = interpro_ids
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_domain", file_path, file_date, source, get_domain_from_uniprot, pid_table)
    return data

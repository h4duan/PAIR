import sys
sys.path.append('..')
from utils import *
import gzip
import xmltodict
import random


# {'@key': '1', 'citation': {'@type': 'submission', '@date': '2003-03', '@db': 'EMBL/GenBank/DDBJ databases', 
# 'title': 'African swine fever virus genomes.', 
# 'authorList': {'person': [{'@name': 'Kutish G.F.'}, {'@name': 'Rock D.L.'}]}}, 
# 'scope': 'NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA]'}
def get_papertitle_from_uniprot(protein):
    content = {}
    paper_titles = []
    #if "dbReference" in df and type(df["dbReference"]) is list:
        #for d in df["dbReference"]:
    if "reference" in protein["entry"]:
        references = protein["entry"]["reference"]
        #print(protein['entry']['accession'])
        #print(references); input()
        if type(references) == dict: references = [references]
        assert type(references) == list
        for reference in references:

            assert 'citation' in reference
            assert type(reference['citation'])==dict
            if 'title' in reference['citation']:
                paper_title=reference['citation']['title']
                paper_info = {'title': paper_title}
                if 'scope' in reference:
                    scopes = reference['scope']
                    if type(scopes) == str: scopes = [scopes]
                    assert type(scopes) == list
                    paper_info['scopes'] = scopes
                paper_titles.append(paper_info)
    #print(paper_titles)
    if len(paper_titles) > 0:
        content = {"text": paper_titles}
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    data = parse_sprot_facts( "protein_papertitle", file_path, file_date, source, get_papertitle_from_uniprot, pid_table)
    return data

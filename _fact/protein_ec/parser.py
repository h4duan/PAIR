import sys
sys.path.append('..')
from utils import *
import utils
import gzip
import xmltodict
import random
from functools import partial

def get_protein_ec_from_uniprot(paths, protein):
    ec_num2text = utils.ec_num_to_text( paths  )
    ec_nums = set()
    df = protein['entry']
    if "dbReference" in df and type(df["dbReference"]) is list:
        for d in df["dbReference"]:
            assert type(d) == dict
            if d['@type'] == "EC":
                ec = d['@id']
                assert type(ec) == str
                if "-" in ec:
                    ec = ec.split("-")[0][:-1]
                    ec = ec.replace("n", "")
                ec_nums.add(ec)
    content = {}
    if len(ec_nums) > 0:
        ec_nums = list(ec_nums)
        ec_texts = []
        for ec in ec_nums:
            if 'n' in ec.split('.')[-1]:
                ec_texts.append( ec_num2text[ '.'.join(ec.split('.')[:-1]) ] )
            else:
                ec_texts.append( ec_num2text[ec] )
        content = { "ec_numbers" : ec_nums, "ec_texts": ec_texts  }
    fact_attr = {}
    if len(content) > 0:
        fact_attr["content"] = content
    return [fact_attr]

def parser(paths, pid_table, cid_table):
    #This fact type uses only uniprot
    file_path, file_date = get_file_path_and_date_from_key( "uniprot-sprot", paths )
    source = "uniprot-sprot"
    get_protein_ec_from_uniprot_with_paths = partial(get_protein_ec_from_uniprot, paths)
    data = parse_sprot_facts( "protein_ec", file_path, file_date, source, get_protein_ec_from_uniprot_with_paths, pid_table )
    return data

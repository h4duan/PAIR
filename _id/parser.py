import sys
import warnings
sys.path.append('..')
import xmltodict
import gzip
from utils import *
from Bio import SeqIO

def get_variant2disease(file_path, date):
    cats = { "LP/P": "This variant is likely pathogenic.", "LB/B": "This variant is likely benign.", "US":  "The signficance of this variant is still uncertain." }
    data = {}
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        lines = []
    for line in lines:
        if "VAR" in line:
            line = line.split()
            if line[4] in cats:
                data[ line[2]  ] = {   "date": date,  "id": line[1] ,  "AAchange": line[3], "category": cats[line[4]], "dbSNP": line[5], "disease": ' '.join( line[6:] )  }
            else:
                print("line error in variants file:", ' '.join(line) )
    return data


def get_isoforms(file_path):
    isoforms = {}
    for record in SeqIO.parse(file_path, "fasta"):
        pid = record.name.split('|')[1]
        original_pid = pid.split('-')[0]
        name = record.name.split('|')[2]
        description = record.description.split('|')[2]
        seq = str(record.seq)
        if original_pid not in isoforms: isoforms[original_pid] = {}
        assert type( name  ) == str
        assert type( description  ) == str
        assert type( seq  ) == str
        assert type( pid  ) == str
        assert type( original_pid  ) == str
        isoforms[original_pid][pid] = { "name": name, "description": description, "aaseq": seq, "variants": []  }
    return isoforms

def get_id_table_from_uniprot(paths, isoforms, variant2disease):
    table = {}
    date = None
    for fin in paths:
        if "uniprot-sprot" in fin:
            file_path = paths[fin]
            date = fin[-7:]
            source = fin
    if date is None: raise Exception( "File for fact type protein_function is not listed!"  )
    with gzip.open(file_path, "rt") as fin:
        i = 0
        entry = ""
        for line in fin:
            entry = line if line.startswith("<entry") else entry + line
            if line.startswith("</entry"):
                protein = xmltodict.parse(entry, xml_attribs=True)
                entry_date = date
                assert type(protein["entry"]["accession"]) == list or type(protein["entry"]["accession"]) == str
                pids = protein["entry"]["accession"]
                if type(pids) == list:
                    pid = pids[0] #other pids are deprecated
                    deprecated_pids = pids[1:]
                else:
                    pid = pids
                    deprecated_pids = None

                organism = {}
                if type(protein["entry"]["organism"]["name"]) != list:
                    protein["entry"]["organism"]["name"] = [protein["entry"]["organism"]["name"]]
                for org in protein["entry"]["organism"]["name"]:
                    organism[org["@type"]] = org["#text"]

                isoform = isoforms[pid] if pid in isoforms else None

                aaseq = protein["entry"]["sequence"]["#text"]

                if "recommendedName" in protein["entry"]["protein"]:
                    name = protein["entry"]["protein"]["recommendedName"]["fullName"]
                elif "submmitedName" in protein["entry"]["protein"]:
                    name = protein["entry"]["protein"]["submmitedName"]
                else:
                    raise Exception("Protein without a name:", pid)
                if type(name) == dict: name = name["#text"] #sometimes there is an envidence number for the name so we have a dict


                if "alternativeName" in protein["entry"]["protein"]:
                    if type(protein["entry"]["protein"]["alternativeName"]) == list: #if there are multiple
                        alternative_names = [i["fullName"]["#text"] if type(i["fullName"]) == dict else i["fullName"] for i in protein["entry"]["protein"]["alternativeName"] ] #sometimes there is an evidence number
                    elif type(protein["entry"]["protein"]["alternativeName"]) == dict: # if there is a single
                        alternative_names = [ protein["entry"]["protein"]["alternativeName"]['fullName']['#text'] ] if type(protein["entry"]["protein"]["alternativeName"]['fullName']) == dict else [ protein["entry"]["protein"]["alternativeName"]["fullName"] ] #sometimes there is an evidence number
                    else:
                        raise Exception( "Unknown type of alternative names for protein:", pid  )
                else:
                    alternative_names = None #some entries do not have alternative names

                DrugBank, STRING = [], []
                if "dbReference" in protein["entry"]:
                    assert type(protein["entry"]["dbReference"]) == list or type(protein["entry"]["dbReference"]) == dict
                    if type(protein["entry"]["dbReference"]) != list: protein["entry"]["dbReference"] = [protein["entry"]["dbReference"]] #if it's a single dict make it a list of dicts
                    for ref in protein["entry"]["dbReference"]:
                        if ref["@type"] == "STRING": STRING += [ref["@id"]]
                        if ref["@type"] == "DrugBank": DrugBank += [ref["@id"]]
                if len(DrugBank) == 0:
                    DrugBank = None
                else:
                    DrugBank = DrugBank # these are ids for facts in DrugBank, so we can have multiple per protein
                if len(STRING) == 0:
                    STRING = None
                elif len(STRING) == 1:
                     STRING = STRING[0]
                else:
                    raise Exception("Multiple STRING ids are being used for protein ", pid, STRING) #we must have at most one STRING id

                variants = []
                if "feature" in protein["entry"]:
                    if type(protein["entry"]["feature"]) == dict:
                        protein["entry"]["feature"] = [protein["entry"]["feature"]]

                    if type(protein["entry"]["feature"]) == list:
                        for feat in protein["entry"]["feature"]:
                            if feat["@type"] == "sequence variant":
                                if "variation" in feat and "original" in feat:
                                    if "@id" not in feat:
                                        feat["@id"] = None
                                    original = feat["original"]
                                    variation = feat["variation"]
                                    var_id = feat["@id"]
                                    var_cat = variant2disease[ var_id  ]["category"] if var_id in variant2disease else None
                                    if "@description" not in feat:
                                        var_descr = None
                                    elif "dbSNP" not in feat["@description"]:
                                        var_descr = feat["@description"]
                                    else:
                                        var_descr = None
                                    if var_id in variant2disease:
                                        var_disease = variant2disease[var_id]["disease"]
                                        if var_disease == "-": var_disease = None
                                    else:
                                        var_disease = None
                                    if len(original) == 1:
                                        start = int(feat["location"]["position"]["@position"]) - 1
                                        end = start + 1
                                    else:
                                        start= int(feat["location"]["begin"]["@position"]) -1
                                        end = int(feat["location"]["end"]["@position"])
                                    if "@sequence" in feat["location"]:
                                        isoform_pid = feat["location"]["@sequence"]
                                        original_seq = isoforms[pid][feat["location"]["@sequence"]]["aaseq"]
                                    else:
                                        isoform_pid = None
                                        original_seq = aaseq
                                    variant = original_seq[:start] + variation + original_seq[end:]
                                    if original_seq[:start] + original + original_seq[end:] != original_seq:
                                        raise Exception("Problem when parsing this variant: "+feat)
                                    if var_disease or var_cat:
                                        entry_date = get_date( [entry_date, variant2disease[var_id]["date"]]  )
                                    if isoform_pid:
                                        if var_id: assert type( var_id  ) == str
                                        assert type( variant  ) == str
                                        isoform[isoform_pid]["variants"] += [ { "id": var_id, "aaseq": variant, "description": var_descr, "disease": var_disease, "category": var_cat  }  ]
                                    else:
                                        variants += [ { "id": var_id, "aaseq": variant, "description": var_descr, "disease": var_disease, "category": var_cat  }  ]
                    else:
                        raise Exception( "Unknown type of protein feature field:", protein["entry"]["feature"]  )
                else:
                    variants = None
                if len(variants) == 0: variants = None

                assert type(aaseq) == str
                assert type(name) == str
                if alternative_names:
                    assert type(alternative_names) == list
                    for i in alternative_names:
                        assert type(i) == str
                if DrugBank:
                    assert type(DrugBank) == list
                    for i in DrugBank:
                        assert type(i) == str
                if STRING:
                    assert type(STRING) == str
                if variants:
                    assert type(variants) == list
                    for i in variants:
                        assert type(i) == dict
                        if i["id"]: assert type(i["id"]) == str
                        assert type(i["aaseq"]) == str

                table[pid] = {  "source": source, "date": entry_date  , "aaseq": aaseq, "name": name, "alternative_names": alternative_names,   "organism": organism,   "deprecated_pids": deprecated_pids, "DrugBank": DrugBank, "STRING": STRING, "variants": variants, "isoforms": isoform  }

                #print(pid, table[pid])

                #print(print_nested_dict(protein))
                #exit()
    return table

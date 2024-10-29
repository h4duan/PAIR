from utils import *
from _id.parser import get_isoforms, get_variant2disease

#Create PID table
paths = yaml.safe_load(open("_config/paths.yml", 'r'))
fact_types = yaml.safe_load(open("_config/fact_types.yml", 'r'))
file_path, file_date = get_file_path_and_date_from_key( "uniprot-isoform", paths )
isoforms = get_isoforms( file_path )
file_path, file_date = get_file_path_and_date_from_key( "variant-data", paths )
variant2disease = get_variant2disease( file_path, file_date  )
from _id.parser import get_id_table_from_uniprot
pid_table = get_id_table_from_uniprot( paths, isoforms, variant2disease )
#compress_json.dump( pid_table, paths["frames"]+"/id_table.json.gz"  )
for ft in fact_types:
    if fact_types[ft]["parse"]:
        print("Parsing", ft, "facts...")
        sys.path.append('_fact')
        ft_parser_module = __import__(ft+".parser")
        sys.path.pop()
        frame = ft_parser_module.parser.parser( paths, pid_table, cid_table )
        print("Saving a total of", len(frame), "facts!")
        compress_json.dump(frame, paths["frames"]+"/protein_"+ ft  +"_frame.json.gz")
print("Parsing finished")

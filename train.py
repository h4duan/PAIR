import pickle
import argparse
from _model import dataloader_ddp
import yaml
import compress_json
import json
from utils import *
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler 
from _model.collate import *
from _model.model import Seq2SeqModel,flatten
from _model.config import get_cfg_defaults
import os
from datetime import datetime
import wandb
from torch import optim, nn
from _model.validation_dataloader import broadcast_and_concatenate_tensor, load_name_tokens, eval_name, eval_ec, TestDataset, load_ec_tokens, eval_go, load_go_tokens, eval_deeploc
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ExponentialLR, CosineAnnealingLR
import pandas as pd
import copy
from torch.cuda.amp import GradScaler, autocast
from _data_split.get_test_pids import expand_test_seqs, load_files, get_temporal_test_pids
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def simple_collate_fn(batch):
    protein_seq = []
    go_seq = []
    pid_seq = []
    for pid, protein, go in batch:
        pid_seq += [pid]
        protein_seq += [protein]
        go_seq.append(go)
    return pid_seq, protein_seq, go_seq


def unique_items_and_counts(strings):
    # Use Counter to count occurrences of each string
    counts = Counter(strings)

    # Extract unique items and their counts
    unique_items = list(counts.keys())
    counts_of_unique_items = list(counts.values())

    return unique_items, counts_of_unique_items

def train_one_epoch(model, prot_loaders, optimizer, scheduler, eval_loader, model_config, all_ec_numbers, all_ec_tokens, max_grad_norms, go_graph, weighted_sample=True, scaler=None, rank = 0):
    model.train(True)
    pids  = []
    texts = []
    current_time = datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset_length = len(prot_loaders.dataset)
    best_loss = 10000
    patience = 0
    best_step = 0
    eval_interval = min(10000, len(prot_loaders))
    if not weighted_sample:
        for data in prot_loaders:
            protein, text = data['x'], data['y']
            pids += data["pid"]
            optimizer.zero_grad()
            loss = model(protein, text)
            loss = torch.mean(loss)
            contains_nan = torch.isnan(loss).any().item()
            if contains_nan:
                print("loss nan")
                print(text)
                wandb.alert(title="OOM error", text="OOM error")
                exit()
            if rank == 0:
                wandb.log({f"training loss": loss.item()})
            loss.backward()
            if model_config.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.train.max_grad_norm)
            optimizer.step()
        if rank == 0:
            save_model(model, optimizer, scheduler, cfg=model_config, time_string=time_string)
        return
    for ii in range(model_config.train.num_itr+1):
        if ii % min(100,eval_interval) == 0:
            prot_loaders.sampler.set_epoch(ii)
            training_loader_iter = iter(prot_loaders)
        data = next(training_loader_iter)
        if not model_config.train.inverse:
            protein, text = data['x'], data['y']
        else:
            text, protein = data['x'], data['y']
        pids += data["pid"]
        optimizer.zero_grad()
        loss = model(protein, text)
        loss = torch.mean(loss)
        contains_nan = torch.isnan(loss).any().item()
        if contains_nan:
            print("loss nan")
            print(text)
            wandb.alert(title="OOM error", text="OOM error")
            exit()
        if rank == 0:
            wandb.log({f"training loss": loss.item()})
        loss.backward()
        if model_config.train.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.train.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if len(eval_loader) == 0 and ii % eval_interval == 0:
            current_time = datetime.now()
            time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            if rank == 0:
                save_model(model, optimizer, scheduler, cfg=model_config, time_string=time_string)
        if len(eval_loader) > 0 and ii > 0 and "lm_loss" in eval_loader and ii % eval_interval == 0:
            eval_loader["lm_loss"].sampler.set_epoch(ii)
            total_eval_loss = torch.zeros(1).to("cuda")
            for eval_data in eval_loader["lm_loss"]:
                eval_protein, eval_text = eval_data["x"], eval_data["y"]
                with torch.no_grad():
                    eval_loss = model(eval_protein, eval_text)
                    eval_loss = torch.mean(eval_loss)
                total_eval_loss += eval_loss
            total_eval_loss = broadcast_and_concatenate_tensor(total_eval_loss / len(eval_loader["lm_loss"]))
            total_eval_loss = torch.mean(total_eval_loss).item()
            if rank == 0:
                wandb.log({f"val loss": total_eval_loss})
            if total_eval_loss < best_loss:
                best_loss = total_eval_loss
                best_step = ii
                patience = 0
                if rank == 0:
                    wandb.log({f"best step": best_step})
                    save_model(model, optimizer, scheduler, cfg=model_config, time_string=time_string)
                #torch.distributed.barrier()
            else:
                patience += 1
                if patience > 3:
                    exit()
            torch.distributed.barrier()
       
        


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    parser = argparse.ArgumentParser(description="Learning Universal Representations for Biochemistry with Deep Neural Networks and Text Supervision")
    parser.add_argument("--parse_id", action="store_true", help="Run the id parser")
    parser.add_argument("--max_buffer_size", type=int, default=5000,  help="Buffer size in dataset")
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="path to .yml file containing config",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        help="path to .yml file containing config",
    )
    parser.add_argument(
        "--multi-fact-ablation",
        dest = "multi_fact",
        action="store_true"
    )
    parser.add_argument(
        "--addition-fact",
        dest = "addition_fact", 
        type=str,
        default=""
    )
    parser.add_argument(
        "--scaling-fact-ablation",
        dest = "scale_fact",
        action="store_true"
    )
    parser.add_argument(
        "--single-fact-ablation",
        dest = "single_fact",
        action="store_true"
    )
    parser.add_argument(
        "--scaling-fact-num",
        dest = "scale_fact_num",
        type=int,
        default=0
    )
    parser.add_argument(
        "--single-fact-num",
        dest = "single_fact_num",
        type=int,
        default=0
    )
    parser.add_argument(
        "--domain-weights",
        dest = "domain_weights",
        type=str,
        default="val"
    )
    parser.add_argument(
        "--num-itr",
        dest = "num_itr",
        type=int,
        default=0
    )
    parser.add_argument(
        "--learning-rate",
        dest = "learning_rate",
        type=float,
        default=0
    )
    parser.add_argument(
        "--use-scheduler",
        dest = "use_scheduler",
        action="store_true"
    )
    parser.add_argument(
        "--encoder-name",
        dest = "encoder_name",
        type=str,
        default=""
    )
    parser.add_argument(
        "--decoder-name",
        dest = "decoder_name",
        type=str,
        default=""
    )
    parser.add_argument(
        "--batch-size",
        dest = "batch_size",
        type=int,
        default=0
    )
    parser.add_argument(
        "--scheduler-gamma",
        dest = "scheduler_gamma",
        type=float,
        default=0
    )
    parser.add_argument(
        "--max-grad-norm",
        dest = "max_grad_norm",
        type=float,
        default=0
    )
 



    
    """ CONFIGURING RUN  """
    args = parser.parse_args()
    paths = yaml.safe_load(open("_config/paths.yml", 'r'))
    fact_types = yaml.safe_load(open("_config/fact_types.yml", 'r'))
    model_config = get_cfg_defaults()
    if args.domain_weights == "uniref50":
        fact2num = {'protein_similarity_text': 160228, 'protein_papertitle': 53666, 'protein_keywords': 145024, 'protein_function': 153617, 'protein_ptm': 19653, 'protein_domain': 177296, 'protein_recommended_name': 198601, 'protein_alternative_name': 103835, 'protein_cofactor': 32812, 'protein_subunit': 88356, 'protein_binding_sites': 61388, 'protein_active_sites': 25339, 'protein_sites': 10603, 'protein_subcellular_location': 130336}
    used_fact = ""
    sorted_facts = sorted(fact2num.keys(), key=lambda k: fact2num[k], reverse=True)
    #print(sorted_facts)
    if args.scale_fact:
        print(f"We study {len(fact2num)} fact types for proteins")
        sorted_facts = sorted(fact2num.keys(), key=lambda k: fact2num[k], reverse=True)
        #print(sorted_facts)
        if args.scale_fact_num > 0:
            used_facts = sorted_facts[:args.scale_fact_num]
        else:
            used_facts = sorted_facts
        print(args.scale_fact_num, len(used_facts))
        for ft in fact_types:
            if ft in used_facts:
                print(ft)
                fact_types[ft]["load"] = True
            else:
                fact_types[ft]["load"] = False
        #exit()
        for ft in fact_types:
            setattr(model_config.fact, ft, fact_types[ft]["load"])
    elif args.single_fact:
        sorted_facts = sorted(fact2num.keys(), key=lambda k: fact2num[k], reverse=True)
        #print(sorted_facts)
        used_fact = sorted_facts[args.single_fact_num]
        print(used_fact)
        #print(args.single_fact_num, len(used_fact))
        for ft in fact_types:
            if ft == used_fact:
                #print(ft)
                fact_types[ft]["load"] = True
            else:
                fact_types[ft]["load"] = False
        #exit()
        for ft in fact_types:
            setattr(model_config.fact, ft, fact_types[ft]["load"])
    elif not args.multi_fact:
        for ft in fact_types:
            setattr(model_config.fact, ft, fact_types[ft]["load"])
            if fact_types[ft]["load"]:
                for key in fact_types[ft]:
                #print(key)
                    setattr(model_config.fact, key, fact_types[ft][key])
    else:
        mila_fact = ["protein_subcellular_location", "protein_recommended_name", "protein_similarity_text", "protein_function"]
        for ft in fact_types:
            if ft in mila_fact:
                fact_types[ft]["load"] = True
            elif ft == args.addition_fact:
                fact_types[ft]["load"] = True
            else:
                fact_types[ft]["load"] = False
        for ft in fact_types:
            setattr(model_config.fact, ft, fact_types[ft]["load"])
    model_config.jobid = os.environ.get('SLURM_JOB_ID')
    if args.num_itr > 0:
        model_config.train.num_itr = args.num_itr
    if args.learning_rate > 0:
        model_config.train.lr = args.learning_rate
    if args.use_scheduler:
        model_config.train.scheduler = True
    if len(args.encoder_name) > 0:
        model_config.train.encoder_name = args.encoder_name
    if args.batch_size > 0:
        model_config.train.batch_size = args.batch_size
    if len(args.decoder_name) > 0:
        model_config.train.decoder_name = args.decoder_name
    if args.scheduler_gamma > 0:
        model_config.train.scheduler_gamma = args.scheduler_gamma
    if args.max_grad_norm > 0:
        model_config.train.max_grad_norm = args.max_grad_norm
    model_config.freeze()
    #print(model_config)
    #exit()
    pid2cluster = None
    uniref50_pid2cluster = None
    """ GETTING GLOBAL PID TABLE FROM UNIPROT  """
    if args.parse_id:
        from _id.parser import get_isoforms, get_variant2disease
        file_path, file_date = get_file_path_and_date_from_key( "uniprot-isoform", paths )
        isoforms = get_isoforms( file_path )
        file_path, file_date = get_file_path_and_date_from_key( "variant-data", paths )
        variant2disease = get_variant2disease( file_path, file_date  )
        from _id.parser import get_id_table_from_uniprot
        pid_table = get_id_table_from_uniprot( paths, isoforms, variant2disease )
        compress_json.dump( pid_table, paths["frames"]+"/id_table.json.gz"  )
    else:
        if model_config.train.eval_only or model_config.train.debug:
            pid_table = json.load(open(paths["frames"]+"id_table_uniref.json"))
        else:
            print("reading uniref50 cluster")
            pid_table_train, pid_table_test, uniref50_pid2cluster, uniref50_cluster2pid = load_files(paths, paths_test)
            pid2cluster = uniref50_pid2cluster
            df_val = pd.read_csv("val_set_mmseq10_uniref50.csv")
            pid_val = df_val["pid"].values
            uniref = []
            for pid in pid_val:
                if pid in uniref50_pid2cluster:
                    uniref += [uniref50_pid2cluster[pid]]
                else:
                    uniref += [""]
            df_val["uniref50"] = uniref
            df_val.to_csv("val_set_mmseq10_uniref50_uniref.csv")
          
            

    """ GETTING GLOBAL CID TABLE FROM PUBCHEM  """
    cid_table = None #This still has to be implemented.

    local_gpu = 0
    rank = 0
    if model_config.train.parallel == "ddp":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        local_gpu = int(os.environ["LOCAL_RANK"])
        #torch.cuda.set_device(local_gpu)
        rank = int(os.environ["RANK"])
        #print(local_gpu, rank)
        torch.cuda.set_device(local_gpu)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        
    """ PARSING FACTS  """
    for ft in fact_types:
        if fact_types[ft]["parse"]:
            print("Parsing", ft, "facts...")
            sys.path.append('_fact')
            ft_parser_module = __import__(ft+".parser")
            sys.path.pop()
            frame = ft_parser_module.parser.parser( paths, pid_table, cid_table )
            print("Saving a total of", len(frame), "facts!")
            compress_json.dump(frame, paths["frames"]+"/protein_"+ ft  +"_frame.json.gz")
    print("Done with parsing!")

    
    """ LOAD EVAL DATALOADER """

    if not (model_config.train.eval_only or model_config.train.debug):
        pid_test = get_temporal_test_pids(pid_table, pid_table_test, uniref50_pid2cluster)
        pids_that_should_not_be_there = expand_test_seqs(pid_test, uniref50_cluster2pid, uniref50_pid2cluster)
    scaler = None
    id_filter = { "pids": set(), "cids": set() }
    eval_loader = {}
    all_ec_tokens = []
    all_ec_numbers = []
    go_graph = None
    val_dataloader = None
    train_pid_table = {}
    train_dataset_pid = pd.read_csv(model_config.train.data_path)
    #print(train_dataset_pid)
    train_pid = set(train_dataset_pid["pid"].values)

    for ppid, entry in pid_table.items():
            #print(entry)
        if ppid in train_pid:
            train_pid_table[ppid] = entry

    if model_config.test.ec:
        all_ec_numbers, ec_trie = load_ec_tokens(paths["frames"], model_config.train.decoder_name)
        ec_validation = TestDataset(validation_dataset, "EC")
        if model_config.test.ec_test_size < len(ec_validation):
            index = random.sample(range(len(ec_validation)), model_config.test.ec_test_size)
            ec_validation = torch.utils.data.Subset(ec_validation, index)

        eval_loader["ec"] = DataLoader(ec_validation, batch_size=model_config.test.ec_batch_size, shuffle=True, collate_fn = simple_collate_fn)
    if model_config.test.name:
        #all_names, name_trie = load_name_tokens(paths["frames"], model_config.train.decoder_name)
        name_validation = TestDataset(validation_dataset, "names")
        if model_config.train.parallel == "ddp":
            sampler = torch.utils.data.distributed.DistributedSampler(name_validation, shuffle=True)
        all_names, name_trie = load_name_tokens(paths["frames"], model_config.train.decoder_name, name_validation.name_preprocessor)
        eval_loader["name"] = DataLoader(name_validation, batch_size=model_config.test.name_batch_size, sampler=sampler, collate_fn = simple_collate_fn)
    if model_config.test.go:
        go_graph = get_go_graph(paths)
        name2term = go_name2term(go_graph)
        #print(name2term)
        #exit()
        all_go_anno, go_trie = load_go_tokens(go_graph, model_config.train.decoder_name, fact_types["protein_go"]["subontology"])
        go_validation = TestDataset(validation_dataset, "GO")
        eval_loader["go"] = DataLoader(go_validation, batch_size=model_config.test.go_batch_size, shuffle=True, collate_fn=simple_collate_fn)

    if model_config.test.deeploc:
        go_deeploc = TestDataset(validation_dataset, "loc")
        eval_loader["deeploc"] = DataLoader(go_deeploc, batch_size=1, shuffle=True)
    if model_config.test.lm_loss:
        val_pid_table = {}
        eval_dataset_pid = pd.read_csv(model_config.test.data_path, header=None)
        eval_pid = set(list(eval_dataset_pid.iloc[:, 0].values))
        #print(len(eval_pid))
        #exit()
        for ppid, entry in pid_table.items():
            #print(entry)
            if ppid in eval_pid:
                val_pid_table[ppid] = entry
        #print(len(val_pid_table))
        #exit()
        id_filter_val = {"pids":set(), "cids":set()}
        val_dataset = dataloader_ddp.BiochemDataset(paths, fact_types, val_pid_table, cid_table, args.max_buffer_size, id_filter_val, domain_weights = fact2num, parallel=model_config.train.parallel, rank = rank)

        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
        eval_loader["lm_loss"] = DataLoader(val_dataset, sampler=sampler, batch_size=model_config.test.lm_batch_size, collate_fn = collate_seq2seq)

    """ CREATE DATALOADER  """
    if not model_config.train.eval_only:

        dataset = dataloader_ddp.BiochemDataset(paths, fact_types, train_pid_table, cid_table, args.max_buffer_size, id_filter, domain_weights = fact2num, parallel=model_config.train.parallel, rank = rank, pid2cluster=pid2cluster)
        print(len(dataset))
    """ CREATE MODEL """
    model = Seq2SeqModel(model_config, local_gpu)

    """ CREATE OPTIMIZER"""
    optimizer = optim.AdamW(
        params=model.model.parameters(),
        lr=model_config.train.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=model_config.train.weight_decay,
    )

    if model_config.train.scheduler:
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 500, eta_min = 5e-5)
        scheduler = StepLR(optimizer,step_size=model_config.train.scheduler_step_size,  gamma=model_config.train.scheduler_gamma)
    else:
        scheduler = None
    if model_config.load_checkpoint:
        checkpoint = torch.load(model_config.checkpoint_path, map_location=torch.device('cpu'))
        if model_config.load_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler != None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    """ Load wandb """
    if rank == 0:
        wandb.login(key="49fcdbf0a6500043ff9402a8555c64a7351364d0")
        name = []
        print(used_fact)
        if model_config.test.ec:
            name += ["ec"]
            if fact_types["protein_ec"]["upsample"]:
                if model_config.train.debug:
                    name += ["upsampleec"]
                else:
                    name += ["upsamplepid"]
            if fact_types["protein_ec"]["joint"]:
                name += ["joint"]
        name += [model_config.train.float_precision]
        if model_config.test.go:
            name += ["go"]
        if model_config.test.name:
            name += ["name"]
        if model_config.test.lm_loss:
            name += ["lm"]
        if model_config.load_checkpoint:
            name += ["fromPretrained"]
        else:
            name += ["randomInitialized"]

        if model_config.train.freeze_encoder:
            name += ["freezeencoder"]
        if model_config.train.freeze_decoder:
            name += ["freezedecoder"]
        if "150" in model_config.train.encoder_name:
            name += ["esm150"]
        if "650" in model_config.train.encoder_name:
            name += ["esm650"]
        if model_config.train.scheduler:
            name += ["lrscheduler"]
        name += [model_config.train.decoder_name]
        name += [f"lr{model_config.train.lr}"]
        name += [f"wd{model_config.train.weight_decay}"] 
        name += [f"dropout{model_config.train.esm_dropout}"]
        wandb.init(project="clip", entity="clip_chem", name="_".join(name))
        wandb.config.update(model_config)

    best_acc = 0
    """ TRAIN  """
    for ii in range(model_config.train.epochs):
        
        #wandb.log({"epochs": ii})
        acc = 0
        #print(ii) 
        if model_config.train.eval_only:
            ii = 1
        #print(len(dataset.buffer))
        if not model_config.train.eval_only and len(dataset.weight) > 0:
            assert len(dataset.weight) == len(dataset.buffer)

            if model_config.train.weighted_sample:
                sampler = WeightedRandomSampler(dataset.weight, len(dataset.buffer), replacement=True)
                if model_config.train.parallel == "ddp":
                    sampler = DistributedSamplerWrapper(sampler, shuffle=True)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)

            dataloader = DataLoader( dataset, batch_size=model_config.train.batch_size, collate_fn = collate_seq2seq, sampler=sampler)
        elif not model_config.train.eval_only:
            #if model_config.train.parallel == "ddp":
            if model_config.train.parallel == "ddp": 
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)

                dataloader = DataLoader(dataset, batch_size=model_config.train.batch_size, collate_fn=collate_seq2seq, sampler=sampler)
            else:
                dataloader = DataLoader(dataset, shuffle=True, batch_size=model_config.train.batch_size, collate_fn=collate_seq2seq)
        
        if not model_config.train.eval_only:
            if model_config.train.parallel == "ddp":
                dataloader.sampler.set_epoch(ii)
            train_one_epoch(model, dataloader, optimizer, scheduler, eval_loader, model_config, all_ec_numbers, all_ec_tokens, model_config.train.max_grad_norm, go_graph, weighted_sample=model_config.train.weighted_sample, rank=rank)

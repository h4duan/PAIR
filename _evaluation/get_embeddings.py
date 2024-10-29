from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, T5Tokenizer, T5Model, T5EncoderModel, OPTModel
from transformers import BertForMaskedLM, BertTokenizer, pipeline
from torch.utils.data import DataLoader
from get_embeddings_haonan import ProtEncoder
import pickle
from transformers import T5Tokenizer, T5EncoderModel, BertModel, BertTokenizer
import esm
from tqdm import tqdm
import torch

import re
import numpy as np
from torch import nn
import argparse
import pandas as pd
#import protst

import os, sys, psutil

cwd = os.getcwd()
utils_dir = "/".join(cwd.split("/")[:-3]) + "/biochem"
sys.path.insert(0, utils_dir)
from utils import reformat_seq
from _model.model import Seq2SeqModel

class ProtEncoder(nn.Module):
    def __init__(self, model_name, model_dtype, embed_dtype, max_length, hidden_layer=-1):
        super(ProtEncoder, self).__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_layer = hidden_layer
        print("loading hidden layer.....", self.hidden_layer)
        if self.model_name == "prot_t5_xl":
            self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50") 
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        elif self.model_name == "prot_bert":
            self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
            self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        elif self.model_name == "prot_albert":
            self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
            self.model = BertModel.from_pretrained("Rostlab/prot_albert")
        elif self.model_name == "GT4SD":
            self.model = T5EncoderModel.from_pretrained('GT4SD/multitask-text-and-chemistry-t5-base-augm')
            self.tokenizer = T5Tokenizer.from_pretrained('GT4SD/multitask-text-and-chemistry-t5-base-augm') 
        elif "esm2" in self.model_name:
            assert "_t" in self.model_name # number of layers
            esm_model_path = "facebook/{}".format(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(esm_model_path)
            self.model = AutoModel.from_pretrained(esm_model_path)
        elif self.model_name == "protst":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            ckpt = torch.load("/work1/maddison/haonand/protst_esm2.pth")
            new_state_dict = {}
            for key, val in ckpt["model"].items():
                if key.startswith("protein_model"):
                    new_state_dict[key[len("protein_model.model."):]] = val
            model.load_state_dict(new_state_dict, strict=True)
            self.model = model
            #self.model = protst.init_model().to('cuda')
        elif self.model_name == "protst1b":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            ckpt = torch.load("/work1/maddison/haonand/protst_esm2.pth")
            new_state_dict = {}
            for key, val in ckpt["model"].items():
                if key.startswith("protein_model"):
                    new_state_dict[key[len("protein_model.model."):]] = val
            model.load_state_dict(new_state_dict, strict=True)
            self.model = model
        elif self.model_name == "galactica":
            self.model = OPTModel.from_pretrained("facebook/galactica-6.7b")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("tokenizer", self.tokenizer.pad_token)
        elif "pair" in self.model_name:
            checkpoint = torch.load(self.model_name, map_location=torch.device('cpu'))
            model_config = checkpoint["cfg"]
            model_config.defrost()
            model_config.train.parallel = "dp"
            model_config.train.num_gpu = 1

            setattr(model_config.train, "freeze_decoder", False)
            setattr(model_config.train, "float_precision", model_dtype)
            setattr(model_config.train, "esm_dropout", 0.0)

            model_config.freeze()
            #loading model
            self.model = Seq2SeqModel(model_config, 0)
            self.tokenizer = self.model.encoder_tokenizer
            self.model= self.model.model.module
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key, val in state_dict.items():
                if "model.module." in key:
                    new_state_dict[key[len("model.module."):]] = val
            self.model.load_state_dict(new_state_dict, strict=True)
            if model_dtype == "bfloat16":
                self.model = self.model.bfloat16()
            elif model_dtype == "float32":
                self.model = self.model.to(torch.float32)
            self.model = self.model.encoder.to("cuda")
        #self.device = "cuda"
        if self.model_name != "protst":
            self.embed_dtype = embed_dtype
            print("model dtype......", self.model.dtype)
            print("embed dtype......", self.embed_dtype)
            print()
    
    def forward(self, sequences):
        #if self.model_name == "protst":
            #sequences = [s.replace("X", "") for s in sequences] 
        #    embeddings = protst.forward(self.model, sequences)
        #    return embeddings, None # format of our return statement
        if self.model_name == "prot_t5_xl": # or self.model_name == "prot_bert":
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
            ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding='max_length', 
                    max_length=self.max_length+1, truncation=True)
        elif self.model_name == "prot_bert" or self.model_name =="prot_albert":
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
            ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding='max_length', 
                    max_length=self.max_length+2, truncation=True)
        elif self.model_name == "galactica":
            ids = self.tokenizer(sequences, return_tensors="pt", max_length=self.max_length+2, truncation=True, return_attention_mask=True)
            ids = ids.to("cuda")
        else:
            ids = self.tokenizer(sequences, return_tensors="pt", padding=True, max_length=self.max_length+2, truncation=True, return_attention_mask=True)
        
    
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
    

        with torch.no_grad():
            if self.model_name == "galactica":
                ids = {key: ids[key] for key in ids if "token_type_ids" not in key}
                outputs = self.model(**ids)
                embedding_repr = outputs.last_hidden_state

            else:
                embedding_repr = self.model(output_hidden_states=True, input_ids=input_ids,attention_mask=attention_mask).hidden_states
                embedding_repr = embedding_repr[self.hidden_layer]
        if "pair" in self.model_name:
            embedding_repr = embedding_repr.to(self.embed_dtype)

        attention_mask = attention_mask.unsqueeze(-1)
        attention_mask = attention_mask.expand(-1, -1, embedding_repr.size(-1))

        if "pair" in self.model_name:
            attention_mask = attention_mask.to(self.embed_dtype)

        # Apply mask
        masked_embedding_repr = embedding_repr * attention_mask
        
        # Compute sum and count of non-zero values
        sum_embedding_repr = masked_embedding_repr.sum(dim=1)
        non_zero_count = attention_mask.sum(dim=1) 
        
        # Compute mean
        mean_embedding_repr = sum_embedding_repr / non_zero_count

        return mean_embedding_repr, embedding_repr

def assert_mean_calc(sequence_examples, model, args):
    sequence_examples = [s[1] for s in sequence_examples]
    mean_embedding_repr, embedding_repr = model(sequence_examples)
    for ii in range(len(sequence_examples)):
        if args.model_name == "prot_t5_xl": # or args.model_name == "prot_bert":
            row_mean = embedding_repr[ii][:len(sequence_examples[ii])+1].mean(dim=0) # prott5 only had eos
        else:
            row_mean = embedding_repr[ii][:len(sequence_examples[ii])+2].mean(dim=0) # esm have bos and eos
        #row_mean = embedding_repr[ii][:len(sequence_examples[ii])].mean(dim=0)
        assert torch.allclose(mean_embedding_repr[ii], row_mean), (mean_embedding_repr[ii], row_mean)

#>>>>>>> 41fc1489ea4089a4d0fa3fe1f6cc2a8128146e4e

def main():
    parser = argparse.ArgumentParser(description="Script to run inference on Enzyme Commision dataset using our model or various baselines.")
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model_name')
    parser.add_argument('--idx', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--hidden_layer', type=int, default=-1)
    parser.add_argument('--embed_dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--model_dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    parser.add_argument("--parse_test", action="store_true")
    parser.add_argument("--parse_deeploc", action="store_true")
    parser.add_argument("--parse_ppi", action="store_true")
    parser.add_argument("--parse_davis_prot", action="store_true")
    parser.add_argument("--parse_bindingdb_prot", action="store_true")
    parser.add_argument("--parse_bindingdb_mol", action="store_true")
    parser.add_argument("--parse_davis_mol", action="store_true")
    parser.add_argument("--parse_deepsol", action="store_true")
    parser.add_argument("--parse_deepsf", action="store_true")
    parser.add_argument("--test_path", type=str, default="/ssd005/projects/uniprot_aspuru/datasets_alllen/test_set_sp202311_spGO.csv")
    parser.add_argument("--val_path", type=str, default="/ssd005/projects/uniprot_aspuru/datasets_alllen/val_set_mmseq10_uniref50.csv")
    parser.add_argument("--train_path", type=str, default="/ssd005/projects/uniprot_aspuru/datasets_alllen/train_set_mmseq10_uniref50.csv")
    parser.add_argument("--num_gpus", type=int, default=4)
    command_run = psutil.Process(os.getpid())
    command_run = " ".join(command_run.cmdline())
    
    args = parser.parse_args()
    print(f"command executed: {command_run}\n")
    print(f"args: {args}\n")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if args.parse_test:
        df = pd.read_csv(args.test_path)
        seq_col='seq'
        id_col='pid'
    elif args.parse_deeploc:
        df = pd.read_csv("/ssd005/projects/uniprot_aspuru/peer_dataset/deeploc.csv")
        seq_col='AAS'
        id_col='UPID'
    elif args.parse_deepsf:
        df = pd.read_csv("/ssd005/projects/uniprot_aspuru/peer_dataset/deepsf_all.csv")
        seq_col='seq'
        id_col='id'
    elif args.parse_deepsol:
        df = pd.read_csv("/ssd005/projects/uniprot_aspuru/peer_dataset/deepsol.csv")
        seq_col='seq'
        id_col='id'
    elif args.parse_davis_prot:
        df = pd.read_csv("/ssd005/projects/uniprot_aspuru/td_davis_cold.csv")
        #df['id'] = df.index
        seq_col='target'
        id_col='target_id'
    elif args.parse_bindingdb_prot:
        df = pd.read_csv("/ssd005/projects/uniprot_aspuru/td_binding_kd_random.csv")
        seq_col='target'
        id_col='target_id'
    elif args.parse_bindingdb_mol:
        df = pd.read_csv("/ssd005/projects/uniprot_aspuru/td_binding_kd_random.csv")
        seq_col='molecule'
        id_col='molecule_id'
    elif args.parse_davis_mol:
        df = pd.read_csv("/ssd005/projects/uniprot_aspuru/td_davis_cold.csv")
        seq_col='molecule'
        id_col='molecule_id'
    elif args.parse_ppi:
#<<<<<<< HEAD
#E        df = pd.read_csv("/work1/maddison/haonand/td_ppi_random_csmapped.csv")
#=======
        #df = pd.read_csv("/ssd005/projects/uniprot_aspuru/td_ppi_random_all.csv")
        df = pd.read_csv("/ssd005/projects/uniprot_aspuru/td_ppi_random_csmapped.csv")
#>>>>>>> 41fc1489ea4089a4d0fa3fe1f6cc2a8128146e4e
        seq_col='seq'
        id_col='id'
    else:
        df1 = pd.read_csv(args.val_path)
        df2 = pd.read_csv(args.train_path)
        df = pd.concat([df1, df2], ignore_index=True, sort=False)
        seq_col='seq'
        id_col='pid'

    if args.parse_ppi:
        print(df)
        print(df.columns)
        df = pd.concat([
                   df[['protein_1', 'protein_2']].melt(value_name=seq_col),
                   df[['protein1_id', 'protein2_id']].melt(value_name=id_col)], axis=1)
        df = df[df[seq_col].notna()]
    if not args.parse_deeploc and not args.parse_deepsf and not args.parse_deepsol and not args.parse_ppi and not args.parse_davis_mol and not args.parse_bindingdb_mol:
        df[seq_col] = df[seq_col].apply(lambda x: reformat_seq(x))

    print(df)

    pid_arr = df[id_col].to_numpy()
    aas_arr = df[seq_col].to_numpy()
    
    if args.embed_dtype == 'float32':
        embed_dtype = torch.float32
    elif args.embed_dtype == 'bfloat16':
        embed_dtype = torch.bfloat16

    data = list(zip(pid_arr, aas_arr))
    print("data length", len(data))
    
    model = ProtEncoder(num_gpus=args.num_gpus, model_name=args.model_name, max_length=args.max_length, embed_dtype=embed_dtype, model_dtype=args.model_dtype, hidden_layer=args.hidden_layer).to("cuda") 
    #model = ProtEncoder(model_name=args.model_name, max_length=args.max_length, embed_dtype=embed_dtype, model_dtype=args.model_dtype, hidden_layer=args.hidden_layer).to("cuda")
    model.eval()

    if args.idx != -1:
        batch_mode = False
        chunk =  np.array_split(np.arange(len(data)), 100)[args.idx]
    else:
        print("loading data in batch mode......")
        batch_mode = True
        data_not_prev_run = []
        for ii in range(len(data)):
            pid, _ = data[ii]
            if os.path.exists(os.path.join(args.save_dir,"{}.pt").format(pid)):
                continue
            data_not_prev_run.append(data[ii])
        print(f"{len(data) - len(data_not_prev_run)} embeds already computed, computing remaining {len(data_not_prev_run)}")
        data = data_not_prev_run

        #if args.model_name != "protst":
        #    assert_mean_calc(data[:16], model, args)

    if batch_mode:
        #data_chunk = [data[i] for i in chunk]
        #if args.parse_ppi:
        #    print("not implemented for ppi"); exit()

        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
        for pids, aas in tqdm(dataloader):
            aas = [aas]
            embeds = model(aas)
            for ii in range(len(pids)):
                torch.save(embeds[ii].detach().cpu(), os.path.join(args.save_dir,"{}.pt").format(pids[ii]))

    else:
        counter = 0
        for ii in tqdm(range(len(chunk))):
            data_idx = chunk[ii]
            sample = data[data_idx]
            pid, aas = sample 
            if args.parse_ppi: 
                #print(aas)
                aas = aas.split("*")
                aas = [seq for seq in aas if seq != ""]
            else:
                aas = [aas]
            if os.path.exists(os.path.join(args.save_dir,"{}.pt").format(pid)):
                continue
            #aas = data[data_idx][1]
            if args.model_name == "protst":
                embeds = []
                for seq in aas:
                    try:
                        embed, _ = model([seq])
                        embeds.append(embed.detach().cpu())
                    except: 
                        counter += 1
                        #print(counter)
                        #input()

                        with open(f"protst_fails_{args.idx}.txt", 'a') as f:
                            f.write(pid + '\n')
                        continue
                if len(embeds) > 0:
                    embeds = torch.cat(embeds, dim=0)
                    torch.save(embeds, os.path.join(args.save_dir,"{}.pt").format(pid))



            else:
                if args.model_name == "galactica":
                    aas = [f"[START_AMINO]{aas[0]}[END_AMINO]"]
                embed, _ = model(aas)
                torch.save(embed.detach().cpu(), os.path.join(args.save_dir,"{}.pt").format(pid))
            #except:
            #    with open(f"protst_fails_{args.idx}.txt", 'a') as f:
            #        f.write(pid + '\n')
            #    continue

if __name__ == "__main__":
    main()
    #sequence_examples = ["PRTEINO", "SEQWENCE"]
    ##model = ProtEncoder(model_name="esm2_650M", max_length=100).to("cuda")
    #model = ProtEncoder(model_name="/ssd005/projects/uniprot_aspuru/protclip_15347_3.pth", max_length=100).to("cuda")
    #model.eval()
    #mean_embedding_repr, embedding_repr = model(sequence_examples)
    #print("MEAN", mean_embedding_repr)
    #print("0", embedding_repr[0, :9].mean(dim=0), embedding_repr[0, :7].shape)
    #print("1", embedding_repr[1, :10].mean(dim=0))

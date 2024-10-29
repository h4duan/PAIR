import torch
import re
import torch.nn.functional as F
from torch import nn
import math
import random
import numpy as np
import wandb
import random as random
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from transformers import AutoModelForCausalLM, T5Tokenizer, T5EncoderModel, GPT2LMHeadModel, AutoModel, EncoderDecoderModel, EncoderDecoderConfig, AutoConfig, BertTokenizer, GPT2Tokenizer, GPT2Model, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.nn.parallel import DistributedDataParallel as DDP
#import torch.nn.parallel.DistributedDataParallel as DDP

def flatten(matrix):
    return [item for row in matrix for item in row]

class Seq2SeqModel(nn.Module):
    def __init__(self, cfg, local_gpu):
        super().__init__()
        self.encoder_name = cfg.train.encoder_name
        self.decoder_name = cfg.train.decoder_name
        if "t5" not in self.encoder_name:
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(cfg.train.encoder_name, cache_dir = "/work1/maddison/haonand")
        else:
            self.encoder_tokenizer = T5Tokenizer.from_pretrained(cfg.train.encoder_name)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(cfg.train.decoder_name)
        self.decoder_tokenizer.eos_token = self.decoder_tokenizer.sep_token
        self.decoder_tokenizer.padding_side = 'right'
        self.encoder_max_len = cfg.train.encoder_max_len
        self.decoder_max_len = cfg.train.decoder_max_len
        if True: 
            if "Rostlab" in self.encoder_name:
                encoder = T5EncoderModel.from_pretrained(self.encoder_name, cache_dir = "/work1/maddison/haonand")
            else:
                encoder = AutoModel.from_pretrained(self.encoder_name, cache_dir = "/work1/maddison/haonand")
            decoder = AutoModelForCausalLM.from_pretrained(self.decoder_name, cache_dir = "/work1/maddison/haonand", is_decoder = True, add_cross_attention = True)
            config_encoder = AutoConfig.from_pretrained(self.encoder_name)
            config_decoder = AutoConfig.from_pretrained(self.decoder_name) 
            config_decoder.is_decoder = True
            config_decoder.add_cross_attention = True
            #@config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
            #config.add_cross_attention = True
            config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
            self.model = EncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)
        elif cfg.train.float_precision == "bfloat16":
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.encoder_name, self.decoder_name, encoder_cache_dir = "/work1/maddison/haonand",decoder_cache_dir = "/work1/maddison/haonand" , decoder_is_decoder=True, add_cross_attention=True, cross_attention_hidden_size = cfg.train.hidden_size, encoder_hidden_dropout_prob=cfg.train.esm_dropout, encoder_attention_probs_dropout_prob=cfg.train.esm_dropout) 
        elif cfg.train.float_precision == "float32":
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.encoder_name, self.decoder_name, decoder_is_decoder=True, cache_dir = "/work1/maddison/haonand", add_cross_attention=True, cross_attention_hidden_size = cfg.train.hidden_size, encoder_hidden_dropout_prob=cfg.train.esm_dropout, encoder_attention_probs_dropout_prob=cfg.train.esm_dropout)
            self.model = self.model.float()
        elif cfg.train.float_precision == "float16":
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.encoder_name, self.decoder_name, decoder_is_decoder=True, add_cross_attention=True, cache_dir = "/work1/maddison/haonand", cross_attention_hidden_size = cfg.train.hidden_size, encoder_torch_dtype=torch.float16, decoder_torch_dtype=torch.float16)
            self.model = self.model.half()
        if cfg.train.freeze_encoder:
            for name, param in self.model.named_parameters():
                
                if name.startswith("encoder"):
                    if cfg.train.unfreeze_last_layer and (name.startswith("encoder.encoder.layer.32") or name.startswith("encoder.encoder.emb_layer_norm")):
                        #print(name)
                        continue
                    param.requires_grad = False 
            #print(name)
        #exit()
        if cfg.train.freeze_decoder:
            for name, param in self.model.named_parameters():
                if name.startswith("decoder"):
                    if (name.startswith("decoder.transformer.h.23")):
                        #print(name)
                        continue
                    param.requires_grad = False
        if cfg.train.freeze_decoder and cfg.train.freeze_encoder:
            for name, param in self.model.named_parameters():
                if not(name.startswith("encoder") or name.startswith("decoder")):
                    param.requires_grad = False
        #exit()
        if cfg.lora.use_lora:
            peft_config = LoraConfig(fan_in_fan_out=True, task_type=TaskType.SEQ_2_SEQ_LM, modules_to_save=["LayerNorm", "emb_layer_norm_after", "enc_to_dec_proj", "crossattention.self.query", "crossattention.self.key", "crossattention.self.value", "crossattention.output.dense"], target_modules=["embeddings.word_embeddings", "embeddings.position_embeddings", "dense", "query", "key", "value"], inference_mode=False, r=cfg.lora.rank, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout) 
        self.num_gpus = cfg.train.num_gpu
        self.device_list = [f"cuda:{ii}" for ii in range(self.num_gpus)]
        self.model.config.decoder_start_token_id = self.decoder_tokenizer.eos_token_id
        #print(self.decoder_tokenizer.pad_token_id, self.decoder_tokenizer.eos_token_id)
        #exit()
        self.model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size 
        if not cfg.train.inverse and cfg.lora.use_lora:
            print("using lora")
            self.model = get_peft_model(self.model, peft_config)
        if cfg.load_checkpoint:
            checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            #if model_config.load_encoder_only:
            for key, val in state_dict.items():
                if "model.module." in key:
                    new_state_dict[key[len("model.module."):]] = val

            self.model.load_state_dict(new_state_dict, strict=True)
            #self.model = self.model.bfloat16()
            if cfg.train.float_precision == "bfloat16":
                self.model = self.model.bfloat16()
            elif cfg.train.float_precision == "float32":
                self.model = self.model.float()
            elif cfg.train.float_precision == "float16":
                self.model = self.model.half()
        #exit()
        self.model = self.model.bfloat16()
        if cfg.train.parallel == "dp":
            self.model = nn.DataParallel(self.model.to(self.device_list[0]), self.device_list)
        else:
            self.model = self.model.to(local_gpu)
            torch.cuda.set_device(local_gpu)
            self.model = DDP(self.model, device_ids=[local_gpu], find_unused_parameters=True)
    def forward(self, protein, text):
        protein = flatten(protein)
        if "Rostlab" in self.encoder_name:
            protein = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein]
        protein_text = list(text)    
        input_tokens = self.encoder_tokenizer(protein, return_tensors="pt", padding=True, max_length = self.encoder_max_len, truncation=True, return_attention_mask=True)
        input_ids = input_tokens.input_ids.to(self.device_list[0])
        input_attentionmask = input_tokens.attention_mask.to(self.device_list[0])
        output_tokens = self.decoder_tokenizer(protein_text, return_tensors="pt", padding=True, max_length = self.decoder_max_len, truncation=True)
        output_ids = output_tokens.input_ids.to(self.device_list[0])
        output_attentionmask = output_tokens.attention_mask.to(self.device_list[0])
        outputs = self.model(input_ids=input_ids, labels=output_ids, attention_mask=input_attentionmask, decoder_attention_mask=output_attentionmask)
        #if random.uniform(0, 1) > 0.1:
        return outputs.loss


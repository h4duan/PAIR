from torch.utils.data import Dataset
import obonet
import sys
import json, compress_json
import pandas as pd
import numpy as np
from collections import Counter
from utils import *
import torch.distributed as dist
import random

def unique_items_and_counts(strings):
    # Use Counter to count occurrences of each string
    counts = Counter(strings)

    # Extract unique items and their counts
    unique_items = list(counts.keys())
    counts_of_unique_items = list(counts.values())

    return unique_items, counts_of_unique_items



class BiochemDataset(Dataset):
    def __init__( self, paths, fact_types, pid_table, cid_table, max_buffer_size, id_filter, parallel="dp", rank=0, pid2cluster=None, domain_weights=None):
        self.path_to_frames = paths["frames"]
        self.fact_types = fact_types
        self.pid_table = pid_table
        self.cid_table = cid_table
        self.max_buffer_size = max_buffer_size
        self.id_filter = id_filter
        self.fact_num = []
        self.parallel = parallel
        self.rank = rank
        self.pid2cluster = pid2cluster
        for ft in self.fact_types:
            self.fact_types[ft]["paths"] = paths
        self.weight = []
        self.domain_weights = domain_weights
        self.name_processor = name_preprocessor(paths["frames"])
        self.sample_buffer()
    
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def sample_buffer(self, epoch=0):
        self.buffer = []
        self.fact_num = []
        self.weight = []
        self.cluster = []
        whole_buffer = []
        clusternum = {}
        for ft in self.fact_types:
            if ft not in self.domain_weights:
                continue
            #print(ft)
            d_weight = self.domain_weights[ft]
            args = self.fact_types[ft]
            #print(args)
            if not args["load"]:
                continue
            sys.path.append('_fact/')
            ft_batch_module = __import__(ft+".batch")
            sys.path.pop()
            frame = compress_json.load(self.path_to_frames+ft+"_frame.json.gz")

            if self.parallel == "ddp":
                fids = list(frame.keys())
                random.Random(epoch).shuffle(fids)
                chunk_size = int(len(list(frame.keys())) // dist.get_world_size())
            else:
                chunk_size = 0
                fids = list(frame.keys())
                random.Random(epoch).shuffle(fids)
            if "name" in ft and "alternative" not in ft:
                batch,cluster = ft_batch_module.batch.get_batch( frame, fids, self.pid_table, self.cid_table, self.max_buffer_size, self.id_filter, name_processor = self.name_processor, args=args, parallel="dp", rank=self.rank, chunk_size=chunk_size)
            else:
                batch,cluster = ft_batch_module.batch.get_batch( frame, fids, self.pid_table, self.cid_table, self.max_buffer_size, self.id_filter, args=args, parallel="dp", rank=self.rank, chunk_size=chunk_size, pid2cluster=self.pid2cluster)
            if self.rank == 0:
                file_object = open("example_annotation.txt", "a")
                print("--------------", file=file_object)
                print(ft, file=file_object)
                print(batch[:5], file=file_object)

            try:
                assert len(batch) == len(cluster)
            except:
                print(ft)
                exit()
            #print(len(batch))
            if len(cluster)  > 0:
                unique_items, counts = unique_items_and_counts(cluster)
                clustercount = dict(zip(unique_items, counts))
                self.weight += [float(d_weight) / float(clustercount[cc]) for cc in cluster]
            else:
                print(ft)
            self.buffer += batch
        self.fact_num += [len(self.buffer)]
        self.weight = np.asarray(self.weight).astype(np.float32)
        assert len(self.weight) == len(self.buffer)
   

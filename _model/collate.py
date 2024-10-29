import torch

#This currently only works for protein subjects, need to extend to mols
def get_xy_seq2seq(example):
    ids = []
    seqs = []
    for key in example:
        if "protein" in key:
            ids.append( key.split('@')[-1]  )
            seqs.append( example[key]  )
    y = example["text@"+'@'.join(ids)]
    return seqs,y, example["pid"]
def collate_seq2seq(data):
    output = { "x": [], "y": [], "pid": []}
    for example in data:
        #print(example)
        x,y, pid = get_xy_seq2seq(example)
        output["x"].append(x)
        output["y"].append(y)
        output["pid"].append(pid)
    return output

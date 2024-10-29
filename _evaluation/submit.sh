#!/bin/bash

#SBATCH -J embedding               # Job name
#SBATCH -o embedding.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 1                  # Total number of mpi tasks requested
#SBATCH -t 48:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi1004x            # Desired partition

# Launch an MPI-based executable
#module load pytorch
source /work1/maddison/haonand/protllm/bin/activate 
modelname="pair.pth"
trainset="../_dataset/train_set.csv"
valset="../_dataset/test_set_sp202401.csv"

if [[ $modelname == *"esm"* ]]; then
	modelpath=$modelname
else
	modelpath="/work1/maddison/haonand/checkpoints/${modelname}"
fi
#echo $modelpath
embeddingpath="/work1/maddison/haonand/${modelname}_embedding"

python3 get_embeddings_haonan.py --batch_size 200 --save_dir $embeddingpath --model_name $modelpath --embed_dtype float32 --model_dtype bfloat16 --hidden_layer -1

python3 knn_embeddings.py --train_path $trainset --val_path $valset --embedding_path $embeddingpath --embedding_path_test $embeddingpath --tasks EC,pfam_domain,family,binding_sites,active_sites,sites,names,spGO



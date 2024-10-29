echo Hostname: `hostname -s`
echo Node Rank ${SLURM_PROCID}
# prepare environment
echo Using Python from: `which python`


if [[ ${SLURM_PROCID} != '0' ]]
then
    echo waiting for 5 seconds for main worker to start first
    sleep 20
fi

NUM_GPUs=4

python -m torch.distributed.launch  --nnodes ${SLURM_NNODES} --node_rank ${SLURM_NODEID} --nproc_per_node ${NUM_GPUs} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} train.py --learning-rate 4e-5 --num-itr 10000 --domain-weights "uniref50" --dist-backend nccl --max_buffer_size 1000000 --scaling-fact-ablation --scaling-fact-num -1

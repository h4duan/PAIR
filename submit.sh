#!/bin/bash

#SBATCH -J ec               # Job name
#SBATCH -o ec.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 8                 # Total number of nodes requested
#SBATCH -n 8                # Total number of mpi tasks requested
#SBATCH -t 48:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi1004x            # Desired partition

# Launch an MPI-based executable
#module load pytorch
source /work1/maddison/haonand/protllm/bin/activate 

MAIN_HOST=`hostname -s`
# this is the current host
export MASTER_ADDR=$MAIN_HOST
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

export NCCL_DEBUG=INFO

/usr/bin/srun --nodes=8 bash -c "bash submit_worker_reference.sh"


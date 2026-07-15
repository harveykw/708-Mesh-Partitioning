#!/bin/bash
#SBATCH --job-name=708
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute

module purge
module load foss


EXE=./partitioner
TRIS=(10000000 20000000 40000000 80000000 160000000)
PROCS=(1 2 4 8 16)

for np in "${PROCS[@]}"; do
  for ntri in "${TRIS[@]}"; do
    srun -n "$np" "$EXE" "$ntri"
  done
done
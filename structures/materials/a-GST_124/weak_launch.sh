#!/bin/bash

# # List of ranks to use
# ranks=(1 2 4 8 16 32 64)
# # ranks=(8)

# Output folder for logs

# ranks=(1 2 4 8 16 32 64)
# rcuts=(4 6)
# overlaps=(0 1)
# nccls=(0 1)
# hidden_dimensions=(128)
# reorders=(0 1)


init_rank=(4)
rcuts=(4.0)
hidden_dimensions=(128)
overlaps=(0)
nccls=(1)
reorders=(0 1)
tiles_x=(1 2 4 8 16)

overlaps_str=("no_overlap" "overlap")
nccls_str=("mpi" "nccl")
reorder_methods=("X" "CUSTOM")


# # Loop through each main file
# for main_file in "${main_files[@]}"; do
#   # Loop through each rank configuration
for tile_x in "${tiles_x[@]}"; do
    for reorder in "${reorders[@]}"; do
        reorder_method=${reorder_methods[$reorder]}
        for hidden_dimension in "${hidden_dimensions[@]}"; do
            for overlap in "${overlaps[@]}"; do
                for nccl in "${nccls[@]}"; do
                    for rcut in "${rcuts[@]}"; do
                        out_folder="weak_${rcut}A_${nccls_str[$nccl]}_${overlaps_str[$overlap]}_${hidden_dimension}hidden_${reorder}order"

                        mkdir -p $out_folder
                        rank=$(( init_rank * tile_x ))
                        # Loop through each rank configuration
                        # Calculate the number of nodes and tasks per node based on the rank
                        if [ "$rank" -le 4 ]; then
                            tasks_per_node=$rank
                            nodes=1
                        else
                            tasks_per_node=4
                            nodes=$(( (rank + 3) / 4 )) # Ceiling division to calculate the number of nodes
                        fi

                        # Create the SLURM script with the appropriate variables
                        cat > slurm_script.sh <<EOL
#!/bin/bash -l
#SBATCH --job-name=${out_folder}_${rank}
#SBATCH --account=lp16
#SBATCH --time=01:00:00
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=$tasks_per_node
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
#SBATCH --output=${out_folder}/output_%j.txt
#SBATCH --error=${out_folder}/error_${rank}.err 

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=29500

conda activate ml

export NCCL_NET='AWS Libfabric'
# export NCCL_CROSS_NIC=1
# export NCCL_NET_GDR_LEVEL=SYS
# export NCCL_SOCKET_IFNAME=hsn
# export FI_CXI_COMPAT=0
# export FI_MR_CACHE_MONITOR=userfaultfd
# export FI_CXI_RX_MATCH_MODE=software
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export FI_CXI_DEFAULT_TX_SIZE=32768
# export FI_CXI_DISABLE_HOST_REGISTER=1
# export NCCL_NCHANNELS_PER_NET_PEER=4


srun --cpu-bind=socket bash -c 'export MPICH_GPU_SUPPORT_ENABLED=1; export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID;
python train_scaling.py -f /capstor/scratch/cscs/amaeder/ -rcut $rcut -nccl $nccl -overlap $overlap -hidden_dim $hidden_dimension -num_epochs 120 -is_reorder $reorder -reorder_method $reorder_method -tile_x $tile_x > ${out_folder}/output_\${SLURM_PROCID}_\${SLURM_NTASKS}.txt'
EOL

                        # Submit the job
                        sbatch slurm_script.sh
                    done
                done
            done
        done
    done
done

# export DEBUG=1;
# nsys profile --force-overwrite true -o "${out_folder}/output_\${SLURM_PROCID}_\${SLURM_NTASKS}"
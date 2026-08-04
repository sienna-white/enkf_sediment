#!/bin/bash
# Job name:
#SBATCH --job-name=RUN_ENKF_FLOCCULATION
#SBATCH --account=fc_anemos #co_aiolos ## << this is the condo that tina bought // could also use fc_anemos#
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
# Wall clock limit (let's set to 10 seconds) 
#SBATCH --time=18:00:00
#
## Commands to run
module load julia

# i=208
# echo "running run $i" 
# cd /global/scratch/users/siennaw/scripts/enkf_sediment
# cp run_floc_model_enkf1.jl copies/run_floc_model_enkf1_$i.jl
# julia -t 20 ./run_floc_model_enkf1.jl 
# julia -t 20 ./run_ens_1D_flocmod.jl  
#  

# / SBATCH --ntasks-per-node=22
# / SBATCH --cpus-per-task=1
julia -t 22 ./run_ens_1D_flocmod.jl

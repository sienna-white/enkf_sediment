#!/bin/bash
# Job name:
#SBATCH --job-name=RUN_ENKF_FLOCCULATION
#
# Account:
#SBATCH --account=fc_anemos #co_aiolos ## << this is the condo that tina bought // could also use fc_anemos#
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1

# Wall clock limit (let's set to 10 seconds) 
#SBATCH --time=10:00:00
#
## Commands to run
module load julia

i=40
echo "running run $i" 
cd /global/scratch/users/siennaw/scripts/enkf_sediment
# cp run_floc_model_enkf1.jl copies/run_floc_model_enkf1_$i.jl

julia -t 10 ./run_box_model.jl  


# 32 >> longer  1e-14 + more noise for parameters (0.0002) >> AWESOME! 
# 34 >> longer  1e-14 +  parameters (0.0001)  >> also good 
# 35>. run 34 for a long time 
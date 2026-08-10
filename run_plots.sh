#!/bin/bash
# Job name:
#SBATCH --job-name=RUN_ENKF_FLOCCULATION
#SBATCH --account=fc_anemos #co_aiolos ## << this is the condo that tina bought // could also use fc_anemos#
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
# Wall clock limit (let's set to 10 seconds) 
#SBATCH --time=00:20:00
#/ SBATCH --takes
#
## Commands to run
module load julia

~/.conda/envs/smoke_env/bin/python -u plot_1D.py & 
~/.conda/envs/smoke_env/bin/python -u plot_1D_mass_flux.py & 
wait 
#!/bin/bash
#SBATCH --job-name=adult_utility_w_queryrewritedefense_w_safety_prompt
#SBATCH --error="/bigtemp/duh6ae/LLM_App_Privacy/src/tests/slurm_scripts/outputs/adult_utility_w_queryrewritedefense_w_safety_prompt_%j.out"
#SBATCH --output="/bigtemp/duh6ae/LLM_App_Privacy/src/tests/slurm_scripts/outputs/adult_utility_w_queryrewritedefense_w_safety_prompt_%j.out"
#SBATCH --nodes=1
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH --time=47:59:59
#SBATCH --nodelist=puma02
#SBATCH --reservation=eda8pc_110

module load cuda-toolkit
module load python
export CUDA_HOME=/sw/ubuntu-22.04/cuda/current
export PATH=/u/duh6ae/.vscode-server/bin/6c3e3dba23e8fadc360aed75ce363ba185c49794/bin/remote-cli:/sw/ubuntu-22.04/slurm/current/bin:/sw/ubuntu-22.04/modules/5.0.1/bin:/sw/ubuntu-22.04/anaconda3/2023.03/bin:/sw/ubuntu-22.04/anaconda3/2023.03/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/opt/puppetlabs/bin:/u/duh6ae/bin:/u/duh6ae/bin:/u/duh6ae/.local/bin
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /bigtemp/duh6ae/LLM_App_Privacy
python -m src.tests.adult_utility_w_queryrewritedefense_w_safety_prompt
#! /bin/sh

#SBATCH --job-name=train_sd_v2
#SBATCH --output=/home/yandex/DL20232024a/ozzafar/discriminative_class_tokens_for_counting/output/trainv2.out # redirect stdout
#SBATCH --error=/home/yandex/DL20232024a/ozzafar/discriminative_class_tokens_for_counting/output/trainv2.err # redirect stderr
#SBATCH --partition=gpu-milo # (see resources section)
#SBATCH --account=gpu-research
#SBATCH --time=720 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines ozzafar changed 1 to 2
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=100000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=2 # GPUs in total

python run.py --train False --evaluate False --experiment True --evaluate_experiment True --is_v2 True

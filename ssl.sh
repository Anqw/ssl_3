#!/usr/bin/env bash
#SBATCH --job-name ssl # CHANGE this to a name of your choice
#SBATCH --partition batch # equivalent to PBS batch
#SBATCH --mail-type=ALL # NONE, BEGIN, END, FAIL, REQUEUE, ALL TIME_LIMIT, TIME$
#SBATCH --mail-user=hujilin@cs.aau.dk # CHANGE THIS to your email address!
#SBATCH --time 21-00:00:00 # Run 24 hours
#SBATCH --qos=deadline # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:2 # CHANGE this if you need more or less GPUs
#SBATCH --nodelist=nv-ai-03.srv.aau.dk # CHANGE this to nodename of your choice$
#SBATCH --mem=300G
#SBATCH --cpus-per-gpu=10


# sinfo
#module load gcc/6.5.0-fxnktbs
#module load cuda/10.0.130-6rlvsy3
srun singularity exec --nv fs3c.sif python setup.py build develop --user || exit 1

srun singularity exec --nv fs3c.sif python tools/train_net.py --num-gpus 2 --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml
srun singularity exec --nv fs3c.sif python tools/ckpt_surgery.py --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth --method randinit --save-dir checkpoints/voc/faster_rcnn/faster_r$
srun singularity exec --nv fs3c.sif python tools/train_net.py --num-gpus 2 --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml




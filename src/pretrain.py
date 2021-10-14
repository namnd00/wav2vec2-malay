# coding:utf-8
"""
Name    : pretrain.py
Author  : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 13/10/2021
Desc:
"""
import argparse
import os
from os.path import join as join_path
import torch
import multiprocessing
import sys


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fairseq_path", default=None, type=str,
                        required=True, help="Path to installed fairseq library")

    parser.add_argument("--audio_path", default=None, type=str,
                        required=True, help="Path to unlabeled audio")

    parser.add_argument("--learning_rate", default=0.005, required=True,
                        type=float, help="Learning rate")
                        
    parser.add_argument("--wandb_project", default="self_supervised_wav2vec2_model", required=False,
                        type=str, help="Weights and Biases project name to use for logging")                    

    parser.add_argument("--init_model", default=None, required=True,
                        type=str, help="Path to pretrain wav2vec model")

    parser.add_argument("--max_tokens", default=480000, required=True,
                        type=int, help="Max of tokens")
                        
    parser.add_argument("--max_epoch", default=100, required=True,
                        type=int, help="Max of epoch")

    parser.add_argument("--config_dir", default=None, required=True,
                        type=str, help="Config dir for training")

    parser.add_argument("--config_name", default=None, required=True,
                        type=str, help="Config name for training")
                        
    parser.add_argument("--results_path", default="./results_path.json", required=False,
                        type=str, help="Results path of eval")

    parser.add_argument("--ext", default="wav", required=False,
                        type=str, help="Extension of wav audio")
                        
    parser.add_argument("--valid_percent", default=0.05, required=False,
                        type=int, help="Validation percent from entire dataset")
                        
    args = parser.parse_args()

    # Prepare manifest file
    MANIFEST_PATH = join_path(args.fairseq_path, "examples/wav2vec/wav2vec_manifest.py")

    temp_dir = os.path.abspath('./temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    cmd = 'python ' + MANIFEST_PATH + ' ' + args.audio_path + ' --dest ' + temp_dir + f' --ext {args.ext} --valid-percent {args.valid_percent}'
    print(cmd)
    print("Prepare training data manifest...")
    os.system(cmd)
    print("End training data manifest.")

    # Pretrain the model
    NUM_GPU = torch.cuda.device_count()
    NUM_CPU = multiprocessing.cpu_count()

    if NUM_GPU == 0:
        print("pytorch cannot find any GPUs !")
        sys.exit(0)

    cmd = ["fairseq-hydra-train"]
    cmd.append("task.data=" + str(temp_dir))
    cmd.append("distributed_training.distributed_world_size=" + str(NUM_GPU))
    cmd.append("+optimization.update_freq='[" + str(int(64 / NUM_GPU)) + "]'")

    if args.init_model is not None:
        cmd.append("checkpoint.restore_file=" + os.path.abspath(args.init_model))
        cmd.append("checkpoint.reset_optimizer=True")
        cmd.append("checkpoint.reset_lr_scheduler=True")
        cmd.append("checkpoint.reset_dataloader=True")
        cmd.append("checkpoint.reset_meters=True")

    cmd.append(f"common.wandb_project={args.wandb_project}")
    cmd.append(f"common_eval.results_path={args.results_path}")
    
    cmd.append(f"optimization.lr=[{args.learning_rate}]")
    cmd.append(f"optimization.max_epoch={args.max_epoch}")
    
    cmd.append(f"dataset.num_workers={NUM_CPU}")
    cmd.append(f"dataset.max_tokens={args.max_tokens}")
    
    cmd.append(f"--config-dir {args.config_dir}")
    cmd.append(f"--config-name {args.config_name}")
    
    cmd = ' '.join(cmd)
    print("Training wav2vec 2.0 large model...")
    os.system(cmd)
    print("End training...")


main()

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

    # parser.add_argument("--manifest_path", default=None, required=True,
    #                     type=str, help="Path to English pretrain wav2vec model")

    parser.add_argument("--init_model", default=None, required=True,
                        type=str, help="Path to pretrain wav2vec model")

    parser.add_argument("--batch_size", default=16, required=True,
                        type=int, help="Batch size, try to decrease this number if any CUDA memory problems occur")

    parser.add_argument("--config_dir", default=None, required=True,
                        type=str, help="Config dir for training")

    parser.add_argument("--config_name", default=None, required=True,
                        type=str, help="Config name for training")

    args = parser.parse_args()

    # Prepare manifest file
    MANIFEST_PATH = join_path(args.fairseq_path, "examples/wav2vec/wav2vec_manifest.py")

    temp_dir = os.path.abspath('./temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    cmd = 'python ' + MANIFEST_PATH + ' ' + args.audio_path + ' --dest ' + temp_dir + ' --ext wav --valid-percent 0.05'
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

    # cmd.append("optimization.max_update=2000000")
    cmd.append("dataset.num_workers=" + str(NUM_CPU))
    cmd.append("dataset.max_tokens=" + str(args.batch_size))
    cmd.append(f"--config-dir {args.config_dir}")
    cmd.append(f"--config-name {args.config_name}")
    cmd = ' '.join(cmd)
    print("Training wav2vec 2.0 large model...")
    os.system(cmd)
    print("End training...")


main()

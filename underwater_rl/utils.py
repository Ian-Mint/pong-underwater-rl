#!/usr/bin/env python3

import argparse
import glob
import multiprocessing as mp
import logging
import logging.handlers
import os
import re
import threading
import time
from typing import Tuple

import cv2
import numpy as np
import torch


def get_logger(store_dir) -> Tuple[logging.Logger, threading.Thread]:
    def log_worker(q):
        while True:
            record = q.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    log_queue = mp.Queue()

    log_path = os.path.join(store_dir, 'output.log')
    logger = logging.Logger('train_status', level=logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter('%(levelname)s | %(processName)s: %(process)d | %(threadName)s | %(message)s'))

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s'))

    qh = logging.handlers.QueueHandler(log_queue)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(qh)

    logger_thread = threading.Thread(target=log_worker, args=(log_queue,), name="LogThread")
    return logger, logger_thread


def models_are_equal(model_1, model_2):
    models_differ = 0
    log = []
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                log.append(key_item_1[0])
            else:
                raise Exception("Models have different structure")
    return models_differ == 0, f"Mismatches found at: {', '.join(log)}"


def convert_images_to_video(image_dir: str, save_dir: str):
    """
    Converts the images in image_dir to an mp4; deletes the original images
    :param image_dir:
    :return:
    """
    assert os.path.isdir(image_dir), f"image directory {image_dir} does not exist"
    assert os.path.isdir(save_dir), f"save directory {save_dir} does not exist"

    filenames = [f for f in glob.glob(os.path.join(image_dir, '*.png'))]
    filenames.sort(key=lambda f: int(re.findall(r'\/(\d+).png', f).pop()))

    assert filenames, f"No png files in {image_dir}"

    height, width, _ = cv2.imread(filenames[0]).shape
    size = (width, height)

    out = cv2.VideoWriter(os.path.join(save_dir, 'pong.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    for f in filenames:
        img = cv2.imread(f)
        out.write(img)
    out.release()


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr


# noinspection PyProtectedMember
def get_args_status_string(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """
    Returns a formatted string of the passed arguments.

    :param parser: The `argparse.ArgumentParser` object to read from
    :param args: The values of the parsed arguments from `parser.parse_args()`
    :return: "--width 160 --height 160 --ball 3.0 ..."
    """
    args_info = parser._option_string_actions
    s = ''
    for k, v in args_info.items():
        if isinstance(v, argparse._StoreAction):
            s += f'{k} {args.__dict__[v.dest]} '
    return s


def tic():
    return time.time()


def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm, (time.time() - tstart)))


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Dynamic Pong RL - utils')

    subparser = parser.add_subparsers(help='Video Converter')
    parser_vc = subparser.add_parser('img2video', help='Convert a directory of png images to mp4 video')
    parser_vc.add_argument('image_directory', help='Image directory')
    parser_vc.add_argument('save_dir', help='Directory for saving pong.mp4')

    args = parser.parse_args()

    if args.image_directory and args.save_dir:
        convert_images_to_video(args.image_directory, args.save_dir)

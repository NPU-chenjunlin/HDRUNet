'''
import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

import numpy as np

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        model.feed_data(data, need_GT=need_GT)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals(need_GT=need_GT)

        sr_img = util.tensor2numpy(visuals['SR'])
        image_path, alignratio_path = util.generate_paths(dataset_dir, img_name)
        util.save_img_with_ratio(image_path, sr_img, alignratio_path)

        logger.info('{:20s}'.format(img_name))
'''
import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

import numpy as np
import torch

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_debug_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)

# === 只取第一张数据 ===
for test_loader in test_loaders:
    data_iter = iter(test_loader)
    data = next(data_iter)   # 只取第一张

    need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
    img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
    img_name = osp.splitext(osp.basename(img_path))[0]
    dataset_dir = osp.join(opt['path']['results_root'], test_loader.dataset.opt['name'])
    util.mkdir(dataset_dir)

    logger.info(f"\n[Step 1] Loaded image: {img_path}, "
                f"shape={data['LQ'].shape}, "
                f"min={torch.min(data['LQ']).item():.4f}, max={torch.max(data['LQ']).item():.4f}")

    # === 前向推理 ===
    model.feed_data(data, need_GT=need_GT)

    logger.info(f"[Step 2] After feed_data, LQ tensor: shape={data['LQ'].shape}, "
                f"dtype={data['LQ'].dtype}, "
                f"min={data['LQ'].min().item():.4f}, max={data['LQ'].max().item():.4f}")

    model.test()
    visuals = model.get_current_visuals(need_GT=need_GT)

    sr_tensor = visuals['SR']
    logger.info(f"[Step 3] SR tensor: shape={sr_tensor.shape}, "
                f"dtype={sr_tensor.dtype}, "
                f"min={sr_tensor.min().item():.4f}, max={sr_tensor.max().item():.4f}")

    sr_img = util.tensor2numpy(sr_tensor)
    logger.info(f"[Step 4] SR numpy: shape={sr_img.shape}, dtype={sr_img.dtype}, "
                f"min={sr_img.min():.4f}, max={sr_img.max():.4f}")

    image_path, alignratio_path = util.generate_paths(dataset_dir, img_name)
    util.save_img_with_ratio(image_path, sr_img, alignratio_path)

    # 保存后再读取确认
    saved_img = util.imread_uint(image_path, 3)
    logger.info(f"[Step 5] Saved output: {image_path}, "
                f"shape={saved_img.shape}, dtype={saved_img.dtype}, "
                f"min={saved_img.min()}, max={saved_img.max()}")

    logger.info(f"Finished single test image: {img_name}")
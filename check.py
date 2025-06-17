import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logger.info("Starting check.py script...")

import lib.sail as sail

import torch


ones = torch.ones(10, 3).float().cuda()
ones[1, 1] = 5.0
reds = torch.ones(10, 3).float().cuda()
sail.point_vis(ones.contiguous().data_ptr(), reds.contiguous().data_ptr(), 10)

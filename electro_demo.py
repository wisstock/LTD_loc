 #!/usr/bin/env python3
""" Copyright © 2021 Borys Olifirov
HEKA .dat file reading test functions with heka_reader module.

https://webdevblog.ru/nasledovanie-i-kompoziciya-rukovodstvo-po-oop-python/

"""

import sys
import os
import logging

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

sys.path.append('modules')
import heka_reader as hr


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)


data_path = os.path.join(sys.path[0], 'data')
raw_rec = hr.Bundle(f'{data_path}/cell1.dat')

series_num = 0
series_count =  len(raw_rec.pul[0])

print(raw_rec.pul[0][4])

for series_num in range(0, series_count):
    logging.info(f'Pulse protocol {raw_rec.pul[0][series_num].Label},  {raw_rec.pul[0][series_num].NumberSweeps} sweeps')
    series_num += 1

fig = plt.figure(figsize=(12, 8))
ax0 = fig.add_subplot()
for i in range(0, raw_rec.pul[0][5].NumberSweeps):
    ax0.plot(raw_rec.data[0, 5, i, 0], alpha=.5)
plt.show()
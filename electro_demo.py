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




all_series_names = [raw_rec.pul[0][i].Label for i in range(0, len(raw_rec.pul[0]))]
logging.info(f'Series in protocol:\n{all_series_names}')

# for series_num in range(0, series_count):
#     logging.info(f'Pulse protocol {raw_rec.pul[0][series_num].Label},  {raw_rec.pul[0][series_num].NumberSweeps} sweeps')
#     series_num += 1

series_num = 5
logging.info(f'Selected series metha:\n{raw_rec.pul[0][series_num]}')

series_time = raw_rec.pul[0][series_num].Time
series_gain = raw_rec.pul[0][series_num].AmplifierState.RealF2Bandwidth
logging.info(f'Series time: {series_time}, gain: {series_gain}')

time_line = np.arange(len(raw_rec.data[0, series_num, 0, 0])) * 5e-5  # (series_time / 1.0e14)
print(time_line)

logging.info(f'Swepps of {raw_rec.pul[0][series_num].Label} series')
fig = plt.figure(figsize=(12, 8))
ax0 = fig.add_subplot()
for i in range(0, raw_rec.pul[0][series_num].NumberSweeps):
    ax0.plot(time_line, raw_rec.data[0, series_num, i, 0] * series_gain, alpha=.5)
plt.show()
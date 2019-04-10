from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np


results = pu.load_results('/home/pulver/Desktop/Experiments/Avoidance/depth/singlecamera/no_big_reward',
                          enable_progress=False,
                          verbose=True)

plt.show(pu.plot_results(results, average_group=True, shaded_err=False))

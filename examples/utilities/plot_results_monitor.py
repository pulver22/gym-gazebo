from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np


results = pu.load_results('/home/pulver/Desktop/avoidance/1/greyscale',
                          enable_progress=False,
                          verbose=True)

plt.show(pu.plot_results(results, average_group=True, shaded_err=False))

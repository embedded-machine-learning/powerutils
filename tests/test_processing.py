# %%
import scipy as sp
from scipy.signal.signaltools import wiener
from pathlib import Path
import pickle
import numpy as np
import annette.hardware.ncs2.parser as ncs2_parser
import annette.utils as utils
import matplotlib.pyplot as plt
from powerutils import processing
power_file= 'vgg16.dat'
power_path = Path('tests','ncs2_test')

"""
rate = 500 # kHz
duration = 169
print('duration',duration)
result = processing.extract_power_profile(power_file, power_path, duration, sample_rate = rate, peak_thresh=0.6,start_thresh=0.7, end_thresh=0.7)

x= np.arange(len(result))/(rate)
plt.plot(x,result)
plt.show()
# %%
"""

power_file= 'vgg16.dat'
power_path = Path('tests','ncs2_test')
latency_file= 'vgg16.csv'
latency_path = Path('tests','ncs2_test')

rate = 500 # kHz
duration = 171

#ncs2_measured = ncs2_parser.extract_data_from_ncs2_report(latency_path, latency_path, latency_file, format="pickle")
#ncs2_results = utils.ncs2_to_format(ncs2_measured)
#duration = np.sum(ncs2_results['measured'])
#print('duration',duration)
ncs2_measured = ncs2_parser.extract_data_from_ncs2_report(latency_path, latency_path, latency_file, format="pickle")
ncs2_results = utils.ncs2_to_format(ncs2_measured)
total_result = processing.extract_power_profile(power_file, power_path, duration, sample_rate = rate)
result = processing.unite_latency_power_meas(ncs2_results, power_file, power_path, sample_rate = 500)
# %%
total_result = total_result*50
print(len(total_result))

x= np.arange(len(total_result))/rate
f = plt.figure()
plt.rcParams["figure.figsize"] = (8,2.5)
plt.plot(x,total_result, label='Power profile')
n = 0
for xc in np.cumsum(ncs2_results['measured']):
    if n == 0:
        plt.axvline(x=xc,c='red',label='Layer transitions')
        n = 1
    else:
        plt.axvline(x=xc,c='red')

plt.xlabel("Time [ms]")
plt.ylabel("Power [W]")

plt.legend()
plt.show()

# %%

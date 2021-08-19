# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def detect_end(data, sample=100, thresh=0.2, vis=False):
    """Detect the first sample in the data .

    Args:
        data ([type]): [description]
        sample (int, optional): Number of samples taken for offset calc. Defaults to 100.
        thresh (float, optional): power threshold. Defaults to 0.2.

    Returns:
        [int]: first sample of the measured network
    """    
    
    min = running_mean(data[:sample],40).min()
    logging.debug("MIN %s" % min)
    data_new = data #- min
    data_rev = np.flip(data_new,0)
    if vis:
        plt.plot(data_rev)
        plt.show()
    end = np.argmax(data_rev > thresh)
    end = len(data_rev)-end
    logging.debug("Last sample %s" % end)
    return end


def detect_start(data, sample=100, thresh=0.1):
    """Detect the first sample in the data .

    Args:
        data ([type]): [description]
        sample (int, optional): Number of samples taken for offset calc. Defaults to 100.
        thresh (float, optional): power threshold. Defaults to 0.1.

    Returns:
        [int]: first sample of the measured network
    """    
    
    min = running_mean(data[:sample],40).min()
    logging.debug("MIN %s" % min)
    data_new = data #- min
    start = np.argmax(data_new > thresh)
    logging.debug("First sample %s" % start)
    return start

def cut_layers(data, offset, times, names, div=500):
    """cut the datasequence into one sequence per layer from offset into offset starting from offset
    -> Returns a dict with the names as keys

    Args:
        data ([type]): [description]
        offset ([type]): [description]
        times ([type]): [description]
        names ([type]): [description]
        div (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """    
    
    # insert offset into the np array
    #get the sample positions
    samples = np.insert( (times*div).astype(int)+offset, 0 ,offset)
    print(samples)
    res = dict()
    for x, name in enumerate(names):
        if samples[x] != samples[x+1]:
            res[name] = data[samples[x]+1:samples[x+1]+1]
        elif x == 0:
            res[name] = data[samples[x]:samples[x]+1]
        else:
            res[name] = np.empty( shape=(0))

    return res
    
def unite_reports(latency, analysis):
    for index, row in latency.iterrows():
        print(row['name'])
        entry = analysis.loc[analysis['LayerName'] == row['name']]
        print(len(entry['GFLOPs']))
        if len(entry) == 1:
            latency.loc[latency['name'] == row['name'],['ops']] = entry['GFLOPs'].to_numpy()
    return latency

def read_analyis_report(report):
    data = pd.read_csv(Path(report), sep=",")
    # rename the column names for better readability and understanding
        # change all "/" and "-" to "_" in LayerName
    for i, elem in enumerate(data["LayerName"]):
        if "/" in elem:
            elem = elem.replace("/", "_")
        if "-" in elem:
            elem = elem.replace("-", "_")
        data["LayerName"][i] = elem
    return data


def extract_power_profile(power_file, power_path, execution_time, sample_rate=500, start_thresh=0.6, end_thresh=0.6, peak_thresh=0.6, multiplier=50., vis=False):
    """AI is creating summary for extract_power_profile

    Args:
        power_file (str): power_file filename
        power_path (str): power_file path
        execution_time (float): execution time in milliseconds
        sample_rate (int, optional): Sample rate in kHz Defaults to 500kHz.

    Returns:
        [np.array]: Numpy array containing the power profile
    """

    downsample = 1
    samplerate = sample_rate/downsample
    data = np.load(Path(power_path, power_file))*multiplier
    data = data[::downsample]
    norm = (data - np.min(data))/np.ptp(data)
    norm = (data )/np.max(data)

    if vis:
        plt.plot(norm)
        plt.show()
    
    start = detect_start(norm,thresh=start_thresh)
    end = detect_end(norm,thresh=end_thresh)
    data2 = data[start:end]
    if vis:
        plt.plot(data2)
        plt.show()

    samples = int(samplerate*execution_time)
    #logging.debug('samples',samples)
    #logging.debug("End", end)
    #logging.debug("start", start)
    new_start = int(end-samples*100)
    new_start = np.max((start,new_start))
    #logging.debug("new start", new_start)
    data2 = norm[new_start:end]
    
    window = running_mean(data2,samples)
    window = (window - np.min(window))/np.ptp(window)
    peaks, l = find_peaks(window, height=peak_thresh, distance=samples)
    logging.debug(l['peak_heights'])
    arr1inds = l['peak_heights'].argsort()
    sort = peaks[arr1inds[::-1]]
    sort_top = sort[:20]
    logging.debug(sort_top)
    logging.debug(peaks)
    logging.debug(samples)
    if vis:
        plt.plot(sort_top, window[sort_top], "x")
        plt.plot(window)
        plt.show()
    if len(peaks) == 0:
        peaks = [np.max(window)]

    ind = -np.min((len(peaks),2))

    von = int(peaks[ind])
    bis = int(peaks[ind]+samples)
    logging.debug(samples)
    #logging.debug(von, bis)
    result = data[new_start+von:new_start+bis]

    return result

def unite_latency_power(layer_times, power_file, power_path, sample_rate = 500, rate_div=1):
    
    # Load power measurements and unify with latency for xilinx format

    data = np.load(Path(power_path, power_file))
    x = np.arange(len(data))

    start = detect_start(data)
    layer_times = ncs2_results
        
    times = np.cumsum(layer_times['measured'].to_numpy())
    names = layer_times['name'].to_numpy()
    res = cut_layers(data, start, times, names)
    mean = {}
    sum = {}

    for key, val in res.items():
        mean[key] = np.mean(val)
        sum[key] = np.sum(val)

    if rate_div is not False:
        print(rate_div)
        for key, val in res.items():
            val = val[::rate_div]
            mean[key] = np.mean(val)
            sum[key] = np.sum(val)
    

    layer_times['mean (V)'] = layer_times['name'].map(mean)
    layer_times['sum (Vs)'] = layer_times['name'].map(sum)
    layer_times['mult (Vs)'] = layer_times['measured']*layer_times['mean (V)']

    return layer_times

def unite_latency_power_meas(layer_times, power_file, power_path, sample_rate = 500, rate_div=1, multiplier=50.):
    
        
    duration = np.sum(layer_times['measured'])
    power = extract_power_profile(power_file, power_path, duration, sample_rate, multiplier=multiplier)
    times = np.cumsum(layer_times['measured'].to_numpy())
    names = layer_times['name'].to_numpy()
    logging.debug("SAMPLES")
    logging.debug(len(power))
    power = power[::rate_div]
    logging.debug(len(power))
    res = cut_layers(power, 0, times, names, sample_rate/rate_div)
    mean = {}
    sum = {}
    length = {}

    for key, val in res.items():
        mean[key] = np.mean(val)
        sum[key] = np.sum(val)
        length[key] = len(val)

    layer_times['mean (V)'] = layer_times['name'].map(mean)
    layer_times['sum (Vs)'] = layer_times['name'].map(sum)/500*rate_div
    layer_times['mult (Vs)'] = layer_times['measured']*layer_times['mean (V)']
    layer_times['samples'] = layer_times['name'].map(length)

    return layer_times, power
# %%

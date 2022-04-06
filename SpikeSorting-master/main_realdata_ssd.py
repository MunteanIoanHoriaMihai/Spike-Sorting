from sklearn.decomposition import PCA

from utils.dataset_parsing.realdata_ssd import parse_ssd_file, find_ssd_files, read_timestamps, separate_by_unit, \
    read_waveforms, get_data_from_unit, plot_sorted_data, plot_sorted_data_all_available_channels, units_by_channel, \
    plot_spikes_on_unit

import numpy as np

DATASET_PATH = './datasets/real_data/M017_2/'

spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
WAVEFORM_LENGTH = 58
TIMESTAMP_LENGTH = 1
NR_CHANNELS = 32
print(f"Number of Units: {spikes_per_unit.shape}")
print(f"Number of Units: {len(unit_electrode)}")
print(f"Number of Spikes in all Units: {np.sum(spikes_per_unit)}")
print(f"Unit - Electrode Assignment: {unit_electrode}")
print("--------------------------------------------")

print(f"DATASET is in folder: {DATASET_PATH}")
timestamp_file, waveform_file, _, _ = find_ssd_files(DATASET_PATH)
print(f"TIMESTAMP file found: {timestamp_file}")
print(f"WAVEFORM file found: {waveform_file}")
print("--------------------------------------------")

timestamps = read_timestamps(timestamp_file)
print(f"Timestamps found in file: {timestamps.shape}")
print(f"Number of spikes in all channels should be equal: {np.sum(spikes_per_unit)}")
print(f"Assert equality: {len(timestamps) == np.sum(spikes_per_unit)}")

timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
print(f"Timestamps per channel should be equal: {list(map(len, timestamps_by_unit))}")
print(f"Assert equality: {list(spikes_per_unit) == list(map(len, timestamps_by_unit))}")
print("--------------------------------------------")

waveforms = read_waveforms(waveform_file)
print(f"Waveforms found in file: {waveforms.shape}")
print(f"Waveforms should be Timestamps*58: {len(timestamps) * WAVEFORM_LENGTH}")
print(f"Assert equality: {len(timestamps) * WAVEFORM_LENGTH == len(waveforms)}")
waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
print(f"Waveforms per channel: {list(map(len, waveforms_by_unit))}")
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
waveform_lens = list(map(len, waveforms_by_unit))
print(f"Waveforms/58 per channel should be equal: {[i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Assert equality: {list(spikes_per_unit) == [i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
print("--------------------------------------------")


plot_spikes_on_unit(waveforms_by_unit, 1, show=True)
plot_spikes_on_unit(waveforms_by_unit, 3, show=True)


units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH)
plot_sorted_data("", units_in_channels[31], labels[31], show=True)
plot_sorted_data_all_available_channels(units_in_channels, labels)
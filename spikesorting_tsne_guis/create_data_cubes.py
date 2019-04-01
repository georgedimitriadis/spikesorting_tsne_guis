
"""
Functions that allow the creation of a (templates x channels x times) average data cube where each (channels x times)
pane is the average of all spikes for that template.

The functions use multiprocessing which in windows cannot be called from another script. So if you are using windows
call the ***_multiprocess functions from the command line (which will run the main() function)

There are two ***_multiprocess functions.
The generate_average_over_spikes_per_template_multiprocess() will look into the results of kilosort
(or spikingcircus after transformed to the kilosort compatible ones) and create the average of all templates that the
sorting algorithm has found (including empty templates and noise templates).

The generate_average_over_spikes_per_template_with_infos_multiprocess will use the spike_info and template_info
dataframes to create the average spike waveform of all templates that are in the template_info df.

"""

import numpy as np
from os import path
from joblib import Parallel, delayed
import sys
from spikesorting_tsne import constants as ct


def load_binary_amplifier_data(file, type=np.int16, number_of_channels=1440):
    raw_extracellular_data = np.memmap(file, mode='r', dtype=type)
    raw_extracellular_data = np.reshape(raw_extracellular_data,
                                        (number_of_channels,
                                         int(raw_extracellular_data.shape[0] / number_of_channels)),
                                        order='F')

    return raw_extracellular_data

'''
# Old non multiprocessing function
def generate_average_over_spikes_per_template(base_folder,
                                              binary_data_filename,
                                              binary_file_datatype,
                                              number_of_channels_in_binary_file,
                                              cut_time_points_around_spike=100):
    channel_map = np.load(path.join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)

    spike_templates = np.load(path.join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(path.join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    spike_times = np.squeeze(np.load(path.join(base_folder, 'spike_times.npy')).astype(np.int))

    num_of_channels = active_channel_map.size

    data_raw_matrix = load_binary_amplifier_data(binary_data_filename, binary_file_datatype,
                                                 number_of_channels_in_binary_file)

    number_of_timepoints_in_raw = data_raw_matrix.shape[1]
    data = np.zeros((number_of_templates, num_of_channels, cut_time_points_around_spike * 2))

    for template in np.arange(number_of_templates):
        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, template))
        spike_times_in_template = np.squeeze(spike_times[spike_indices_in_template])
        num_of_spikes_in_template = spike_indices_in_template.shape[0]
        y = np.zeros((num_of_channels, cut_time_points_around_spike * 2))
        if num_of_spikes_in_template != 0:
            # remove any spikes that don't have enough time points
            too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < cut_time_points_around_spike), axis=1)
            too_late_spikes = np.squeeze(np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - cut_time_points_around_spike), axis=1)
            out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))
            spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)
            num_of_spikes_in_template = spike_indices_in_template.shape[0]

            for spike_in_template in spike_indices_in_template:
                y = y + data_raw_matrix[active_channel_map,
                                        spike_times[spike_in_template] - cut_time_points_around_spike:
                                        spike_times[spike_in_template] + cut_time_points_around_spike]

            y = y / num_of_spikes_in_template
        data[template, :, :] = y
        del y
        print('Added template ' + str(template) + ' with ' + str(num_of_spikes_in_template) + ' spikes')

    np.save(path.join(base_folder, 'avg_spike_template2.npy'), data)
'''


# ---------------------------------------------------------------------------
# Functions to create the average from the kilosort results
# ---------------------------------------------------------------------------


def _avg_of_single_template(template,
                            spike_times,
                            spike_templates,
                            num_of_channels,
                            cut_time_points_around_spike,
                            number_of_timepoints_in_raw,
                            data_raw_matrix,
                            active_channel_map):

    spike_indices_in_template = np.argwhere(np.in1d(spike_templates, template))
    spike_times_in_template = np.squeeze(spike_times[spike_indices_in_template])
    num_of_spikes_in_template = spike_indices_in_template.shape[0]
    y = np.zeros((num_of_channels, cut_time_points_around_spike * 2))
    if num_of_spikes_in_template != 0:
        # remove any spikes that don't have enough time points
        too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < cut_time_points_around_spike), axis=1)
        too_late_spikes = np.squeeze(np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - cut_time_points_around_spike), axis=1)
        out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))
        spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)
        num_of_spikes_in_template = spike_indices_in_template.shape[0]

        for spike_in_template in spike_indices_in_template:
            y = y + data_raw_matrix[active_channel_map,
                                    spike_times[spike_in_template] - cut_time_points_around_spike:
                                    spike_times[spike_in_template] + cut_time_points_around_spike]

        y = y / num_of_spikes_in_template
        print('Added template ' + str(template) + ' with ' + str(num_of_spikes_in_template) + ' spikes')
    return template, y


def generate_average_over_spikes_per_template_multiprocess(base_folder,
                                                           binary_data_filename,
                                                           binary_file_datatype,
                                                           number_of_channels_in_binary_file,
                                                           cut_time_points_around_spike=100):
    channel_map = np.load(path.join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)

    spike_templates = np.load(path.join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(path.join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    spike_times = np.squeeze(np.load(path.join(base_folder, 'spike_times.npy')).astype(np.int))

    num_of_channels = active_channel_map.size

    data_raw_matrix = load_binary_amplifier_data(binary_data_filename, binary_file_datatype,
                                                 number_of_channels_in_binary_file)

    number_of_timepoints_in_raw = data_raw_matrix.shape[1]

    unordered_data = Parallel(n_jobs=30)(delayed(_avg_of_single_template)(i,
                                                                         spike_times,
                                                                         spike_templates,
                                                                         num_of_channels,
                                                                         cut_time_points_around_spike,
                                                                         number_of_timepoints_in_raw,
                                                                         data_raw_matrix,
                                                                         active_channel_map)
                                        for i in np.arange(number_of_templates))
    data = np.zeros((number_of_templates, num_of_channels, cut_time_points_around_spike * 2))
    for idx, info in unordered_data:
        data[idx, ...] = info

    np.save(path.join(base_folder, 'avg_spike_template.npy'), data)


# ---------------------------------------------------------------------------
# Functions to create the average from the template_info and spike_info files
# ---------------------------------------------------------------------------


def _avg_of_single_template_with_infos(template_index,
                                       spike_info,
                                       template_info,
                                       cut_time_points_around_spike,
                                       number_of_timepoints_in_raw,
                                       data_raw_matrix,
                                       active_channel_map):

    spike_indices_in_template = template_info['spikes in template'].iloc[template_index]
    template = template_info['template number'].iloc[template_index]

    num_of_spikes_in_template = spike_indices_in_template.shape[0]

    spike_times_in_template = spike_info[spike_info[ct.ORIGINAL_INDEX].isin(spike_indices_in_template)][ct.TIMES]

    if num_of_spikes_in_template != 0:
        # remove any spikes that don't have enough time points
        too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < cut_time_points_around_spike), axis=1)
        too_late_spikes = np.squeeze(np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - cut_time_points_around_spike), axis=1)
        out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))

        spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)

        spike_times = spike_info[spike_info[ct.ORIGINAL_INDEX].isin(spike_indices_in_template)][ct.TIMES].values
        spike_intervals = [(np.arange(spike_time - cut_time_points_around_spike,
                                      spike_time + cut_time_points_around_spike)).astype(np.int32)
                           for spike_time in spike_times]

        data = data_raw_matrix[:, spike_intervals].astype(np.float32)
        data = np.mean(data, axis=1)

        data = data[active_channel_map, :]
        print('Added template ' + str(template) + ' with ' + str(spike_indices_in_template.shape[0]) + ' spikes')
    return template_index, data


def generate_average_over_spikes_per_template_with_infos_multiprocess(base_folder,
                                                                      spike_info,
                                                                      template_info,
                                                                      binary_data_filename,
                                                                      binary_file_datatype,
                                                                      number_of_channels_in_binary_file,
                                                                      cut_time_points_around_spike=100):
    channel_map = np.load(path.join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)
    num_of_channels = len(active_channel_map)

    number_of_templates = template_info.shape[0]

    data_raw_matrix = load_binary_amplifier_data(binary_data_filename, binary_file_datatype,
                                                 number_of_channels_in_binary_file)

    number_of_timepoints_in_raw = data_raw_matrix.shape[1]

    unordered_data = Parallel(n_jobs=12)(delayed(_avg_of_single_template_with_infos)(template,
                                                                                    spike_info,
                                                                                    template_info,
                                                                                    cut_time_points_around_spike,
                                                                                    number_of_timepoints_in_raw,
                                                                                    data_raw_matrix,
                                                                                    active_channel_map)
                                         for template in np.arange(number_of_templates))

    data = np.zeros((number_of_templates, num_of_channels, cut_time_points_around_spike * 2)).astype(np.float32)
    for idx, info in unordered_data:
        data[idx, ...] = info

    np.save(path.join(base_folder, 'avg_spike_template.npy'), data)


# ---------------------------------------------------------------------------
# The main function for calling everything form the cmd
# ---------------------------------------------------------------------------


def main(args):
    if args[1] == 'original':
        base_folder = args[2]
        binary_data_filename = args[3]
        number_of_channels_in_binary_file = int(args[4])
        cut_time_points_around_spike = int(args[5])
        generate_average_over_spikes_per_template_multiprocess(base_folder=base_folder,
                                                               binary_data_filename=binary_data_filename,
                                                               binary_file_datatype=np.int16,
                                                               number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                                                               cut_time_points_around_spike=cut_time_points_around_spike)

    elif args[1] == 'infos':

        base_folder = args[2]
        binary_data_filename = args[3]
        spike_info_filename = args[4]
        template_info_filename = args[5]
        number_of_channels_in_binary_file = int(args[6])
        cut_time_points_around_spike = int(args[7])

        binary_file_datatype = np.int16

        spike_info = np.load(spike_info_filename)
        template_info = np.load(template_info_filename)

        cut_time_points_around_spike = cut_time_points_around_spike

        generate_average_over_spikes_per_template_with_infos_multiprocess(base_folder,
                                                                          spike_info,
                                                                          template_info,
                                                                          binary_data_filename,
                                                                          binary_file_datatype,
                                                                          number_of_channels_in_binary_file,
                                                                          cut_time_points_around_spike)


if __name__ == "__main__":
    args = sys.argv
    main(args)
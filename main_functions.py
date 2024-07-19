from utils import *


def impedance_modulation(main_opt_params, main_proc_params):
    ### Load data ###
    path_channels_quantiles = "Dataset/ChQ.mat"
    channels_quantiles = load_matlab_file(path_channels_quantiles, verbose=False)

    for noise_type in main_proc_params['noise_type']:
        for r in main_proc_params['r']:
            for cost_fcn in main_proc_params['cost_functions']:
                results_dict = {}
                results_dict = impedance_modulation_SNR(main_proc_params, main_opt_params, channels_quantiles,
                                                        cost_fcn, main_proc_params['test_size_single_channel'],
                                                        compute_ml=True, results_dict=results_dict,
                                                        noise_type=noise_type, r=r)
                saving_path = "Impedance_BER/SingleChannel"
                save_data_and_figures(saving_path, cost_fcn, results_dict, mode="single", noisy=main_proc_params['noisy'], noise_type=noise_type, r=r)




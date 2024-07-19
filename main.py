from main_functions import *
import argparse

if __name__ == '__main__':

    # Define input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', help='Noise type', default=True)
    parser.add_argument('--noise_type', help='Noise type', default="symm")
    parser.add_argument('--noise_rate', help='Noise rate', default=0.1)
    parser.add_argument('--learning_rate', help='Learning rate of the optimizer for the neural network',
                        default=0.0001)
    parser.add_argument('--alpha', default=1)
    parser.add_argument("-f", "--file", required=False)
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main_opt_params = {
        'learning_rate': float(args.learning_rate)
    }

    main_proc_params = {
        'SNR_vec': range(-5, 35),
        'M': 4,
        'cost_functions': [2,3,5],
        'alpha': float(args.alpha),
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        'random_seed': 1,
        'test_size_single_channel': 10000,
        'noisy': bool(args.noisy),
        'noise_type': [str(args.noise_type)], #["symm", "sparse", "unif"],
        'r': [float(args.noise_rate)], #[0.1, 0.2],
        'single_channel': True
    }

    print('Selected device: {}'.format(main_proc_params['device']))
    impedance_modulation(main_opt_params, main_proc_params)


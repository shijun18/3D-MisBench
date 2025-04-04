import re
import numpy as np

def find_times(log_file):

    with open(log_file, 'r') as file:
        log_content = file.readlines()

    epoch_times = []

    # read log file
    for line in log_content:

        matches = re.findall(r'Epoch time: (\d+\.\d+) s', line)
        
        epoch_times.extend([float(match) for match in matches])

    if epoch_times:
        return epoch_times
        
        # average_time = sum(epoch_times) / len(epoch_times)
        
        # return average_time
    else:
        raise('error!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str,required=True,
                        help="path to the log file ")
    args = parser.parse_args()
    
    a_0 = find_times(args.log_file)
    # define the start and end epoch indices
    start = 100
    end = 150
    times = a_0[start:end]
    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f'avg: {avg_time:.4f}秒')
    print(f'std: {std_time:.4f}秒')
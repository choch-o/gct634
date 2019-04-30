import os
from multiprocessing import Pool

def run_process(process):
    os.system ('python {}'.format(process))

if __name__ == '__main__':
    processes = []
    for i in range(8):
        processes.append('train_test.py --device=' + str(i+1))
    pool = Pool(processes=8)
    pool.map(run_process, processes)

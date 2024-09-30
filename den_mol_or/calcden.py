import subprocess
from multiprocessing import Process, Queue
import sys
import os
import iodens

def calculate_mult(jobs, n_procs):
    procs = []
    q = Queue()
    for job in jobs:
        print(job)
        proc = Process(target = iodens.gennpz_mwfn, name = job, args = (path + "/" + job,))
        procs.append(proc)

    p_num = len(procs) - 1
    while True:
        liv = 0
        for i in procs:
            if i.is_alive():
                liv += 1

        if liv <= n_procs:
            for k in range(0, n_procs - liv):
                procs[p_num].start()
                p_num -= 1
                if p_num == -1:
                    break
        if p_num == -1:
            break

    for i in procs:
        i.join()

    return 1

path = sys.argv[1]
files = os.listdir(path)

print(files)
ans = calculate_mult(files, int(sys.argv[2]))

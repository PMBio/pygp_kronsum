from subprocess import call
import os
import pdb
from run_simulationsRUNTIME import measure_runtime
import sys

if __name__ == "__main__":
    n_repeats = 10
    N = int(sys.argv[1])
    
    if len(sys.argv)>2:
        n_repeats = int(sys.argv[2])
    
    env = {'sim_dir':os.path.join('..','sim_data'),'out_dir':os.path.join('..','out')}
    #for N in [16, 32, 64, 128, 256]:
    for D in [16, 32, 64, 128, 256]:
        print N,D
        measure_runtime(env,N,D,n_reps=n_repeats)
      

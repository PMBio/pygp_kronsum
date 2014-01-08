from subprocess import call
import os
import pdb
from run_simulationsHIDDEN import run_simulationsHIDDEN
import sys 

if __name__ == "__main__":
    if len(sys.argv)>2:
        n_repeats= int(sys.argv[2])
    else:
        n_repeats = 30
        
    methods = 'ALL'
    env = {'sim_dir':os.path.join('..','sim_data'),'out_dir':os.path.join('..','out')}

    if sys.argv[1]=='common':
        for f_causal in [0.9]:
            for f_common in [0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
                for f_hidden in [0.5]:
                    run_simulationsHIDDEN(env,f_causal,f_common,f_hidden,methods,n_repeats)


    if sys.argv[1]=='causal':
        for f_causal in [0.1,0.3,0.5,0.7,0.9,1.0]:
            for f_common in [0.9]:
                for f_hidden in [0.5]:
                    run_simulationsHIDDEN(env,f_causal,f_common,f_hidden,methods,n_repeats)

    if sys.argv[1]=='hidden':
        for f_causal in [0.9]:
            for f_common in [0.9]:
                for f_hidden in [0.0,0.1,0.3,0.5,0.7,0.9,1]:
                    run_simulationsHIDDEN(env,f_causal,f_common,f_hidden,methods,n_repeats)
  
      



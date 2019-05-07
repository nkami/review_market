'''
  File:   run_experiment.py
  Author: Nicholas Mattei (nsmattei@gmail.com)
  Date: Feb 12 2019

  About
  --------------------
    This is a driver file for the Envy Experiments for capicated discrete allocation 
    paper.  Given a directory it runs the required python algorithms for all variants an
    saves the pickle files in a sub-directory.

    Note: This is likely not the best way to do this but 
    it's good enough for now...

'''


import os
import glob

if __name__ == "__main__":

  # # PREFLIB DATA BLOCK
  # DATA_DIRECTORY = "../dat/preflib_small/input_files/"
  # OUT_DIRECTORY = "../dat/preflib_small/pickle_files/"
  # FILE_TYPE = "*.toi"
  # AGENT_MINMAX = (4,7)
  # OBJECT_MINMAX = (3,4)

  # GENERATED DATA BLOCK
  PHI = 95
  DATA_DIRECTORY = "../dat/ijcai2019_strict_data/phi-" + str(PHI) + "/input_files/"
  OUT_DIRECTORY = "../dat/ijcai2019_strict_data/phi-" + str(PHI) + "/pickle_files/"
  FILE_TYPE = "*.soc"
  AGENT_MINMAX = (3,6)
  OBJECT_MINMAX = (3,4)

  ALL_OBJECTIVES = "-u -e -r -l -c -m -n"
  # ALL_OBJECTIVES = "-u -r"
  # ALL_OBJECTIVES = "-u"
  run_cmd = '''python3 ./cap_discrete_alloc.py'''
  params = ' -a {} -A {} -o {} -O {}'.format(AGENT_MINMAX[0],AGENT_MINMAX[1],OBJECT_MINMAX[0],OBJECT_MINMAX[1])

  # Iterate and run.
  for filepath in sorted(glob.iglob(DATA_DIRECTORY + FILE_TYPE)):
    print("Processing: ",filepath)
    fname = filepath.split("/")[-1]
    outpath = OUT_DIRECTORY + fname[:-4] + '_result.pickle'

    os.system("{} -d {} -p {} {} {}".format(run_cmd, filepath, outpath, params, ALL_OBJECTIVES))

    





'''
  File:   generate_strict_preferences.py
  Author: Nicholas Mattei (nsmattei@gmail.com)
  Date: Feb 12 2019

  About
  --------------------
    Driver file that generates a bunch of random data for testing.


'''

from preflibtools import io
from preflibtools import generate_profiles


NUM_VOTERS = 10
NUM_CANDS = 20
PHI = 95


OUT_PATH = "../dat/ijcai2019_strict_data/phi-" + str(PHI) + "/input_files/"
BASE_FILENAME = "Mallows-" + str(PHI) + "-"
NSAMPLES = 10

for n in range(NSAMPLES):

  cmap = generate_profiles.gen_cand_map(NUM_CANDS)
  rmaps, rmapscounts = generate_profiles.gen_mallows(NUM_VOTERS,
                generate_profiles.gen_cand_map(NUM_CANDS),
                [1], [PHI/100.], [list(sorted(cmap.keys()))])

  outf = open(OUT_PATH + BASE_FILENAME + str(n) + ".soc", 'w')
  io.write_map(cmap, NUM_VOTERS, generate_profiles.rankmap_to_voteset(rmaps, rmapscounts),outf)
  outf.close()

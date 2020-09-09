'''
	Temp script to run set of experiments.
'''

import os
import glob

if __name__ == "__main__":
	#path = '''./json/for\\ AAAI21\\ submission/AAAI17_sim/'''
	files = [ 	"./json/for_AAAI21_submission/AAAI17_sim/AAAI17_600X400_orig_U.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_613X201_orig_U.json",
				"./json/for_AAAI21_submission/ICLR_sim/ICLR_600X400_uniform_U.json",
				"./json/for_AAAI21_submission/ICLR_sim/ICLR_900X300_uniform_U.json",
				
	         ]
	run_cmd = "python3 ./simulation.py"
	for file in files:
		print("Processing: ",file)
		os.system(run_cmd + " " + file)

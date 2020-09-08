'''
	Temp script to run set of experiments.
'''

import os
import glob

if __name__ == "__main__":
	#path = '''./json/for\\ AAAI21\\ submission/AAAI17_sim/'''
	files = [ 	"./json/for_AAAI21_submission/AAAI17_sim/AAAI17_1200X300_orig.json",
			  	"./json/for_AAAI21_submission/AAAI17_sim/AAAI17_2000X200_orig.json",
				"./json/for_AAAI21_submission/AAAI17_sim/AAAI17_600X400_orig.json",
				"./json/for_AAAI21_submission/ICLR_sim/ICLR_150X100_greedy.json",
				"./json/for_AAAI21_submission/ICLR_sim/ICLR_150X100_uniform.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_176X146_orig.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_176X146_uniform.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_442X161_greedy.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_442X161_orig.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_442X161_uniform.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_613X201_greedy.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_613X201_orig.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_613X201_uniform.json"
	         ]
	run_cmd = "python3 ./simulation.py"
	for file in files:
		print("Processing: ",file)
		os.system(run_cmd + " " + file)

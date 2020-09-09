'''
	Temp script to run set of experiments.
'''

import os
import glob

if __name__ == "__main__":
	#path = '''./json/for\\ AAAI21\\ submission/AAAI17_sim/'''
	files = [   "./json/for_AAAI21_submission/PrefLib_sim/PrefLib_52X24_greedy.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_52X24_orig.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_52X24_uniform.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_54X31_greedy.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_54X31_orig.json",
				"./json/for_AAAI21_submission/PrefLib_sim/PrefLib_54X31_uniform.json",
				
	         ]
	run_cmd = "python3 ./simulation.py"
	for file in files:
		print("Processing: ",file)
		os.system(run_cmd + " " + file)

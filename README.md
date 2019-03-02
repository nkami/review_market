run requirements:
-python 3.6
-numpy library installed (version 1.15.3)

usage instructions:
1. put an input preflib file inside the directory "review_market" (the directory where you extracted the zip file)
2. type in the command line "python Match.py $r $name_of_file $mechanism"
$r = is an integer argument that determines how many time each paper need to be reviewed (right now all the are reviewed the same amount of times)
$name_of_file = is the name of the input preflib file
$mechanism = determines which mechanism is used to update the papers prices.
	right now there are 2 mechanisms: 
	-"FixedMechanism" which sets all the prices to 1 and the threshold to k
	-"PriceMechanism" which sets all the prices to 1 and the threshold to k. after a reviewer bids the prices are updated to be min{1, r/demand}
3. an output text file named "simulation_output" will be created in "review_market" directory.

The file will consist of three allocation matrices from steps 1-3 of the Fractional Allocation Algorithm.

example:
>python Match.py 2 example.txt FixedMechanism


usage instructions:
1. put an input preflib file inside the directory "review_market" (the directory where all the files are)
2. type in the command line "python Match.py $r $name_of_file $mechanism" <br/> 
$r = is an integer argument that determines how many times each paper needs to be reviewed (right now all the papers are reviewed the same amount of times) <br/>
$name_of_file = is the name of the input preflib file <br/>
$mechanism = determines which mechanism is used to update the papers prices. <br/>
	right now there are 2 mechanisms: <br/>
	-"FixedMechanism" which sets all the prices to 1 and the threshold to k <br/>
	-"PriceMechanism" which sets all the prices to 1 and the threshold to k. after a reviewer bids the prices are updated to be min{1, r/demand} <br/>
3. an output text file named "simulation_output" will be created in "review_market" directory.

The file will consist of three allocation matrices from steps 1-3 of the Fractional Allocation Algorithm.

example:
>python Match.py 2 example.txt FixedMechanism

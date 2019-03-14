
**Usage Instructions**

1. In the directory of the python files add the path ./data/PrefLib. Put all the desired PrefLib files in the PrefLib directory.
2. Run from terminal the command: python Match.py $input_file <br/>
	2.1 $input_file is the path to the input file (not the PrefLib file) for the simulation. <br/>
	2.2 The input file is a text file with the format: <br/>
	
		<Data processing>
		PreferenceFile: name_of_the_PrefLib_file.txt
		UnallocatedPapersPrices: float
		PrivatePricesGenerators: PrivatePricesGenerator

		<Simulation> 
		Algorithms: matching_algorithm
		MarketMechanisms: market_mechanism
		ReviewRequirements: positive_integer

		<Behavior> 
		Behaviors: behavior
		
	2.3 Its possible to run a few different simulations with the same random bidding order and private prices if one of the 	parameters in the input file has a few arguments seperated by ', '. (e.g. MarketMechanisms: MixedMechanism, FixedMechanism 	   wil run 2 simulations with the different mechanisms).
3. The output will be written as a csv file to a new directory named "output".
	
**Parameters**

PreferenceFile: 

	a name of a PrefLib file in the PrefLib directory with '.txt' added.

UnallocatedPapersPrices: 

	float

PrivatePricesGenerators:

	SimplePrivatePricesGenerator - Assign a random private cost to each paper, while satisfying 
	the condition that every paper with preference rank i > j will have lower cost than every 
	paper with preference rank j (i.e. papers that are more desirable according to preference will 
	have a lower cost)

Algorithms:

    FractionalAllocation - A detailed explanation of this algorithm is available in the review_market.new2
    pdf, pages 5-7.

MarketMechanisms:

	FixedMechanism - Sets all prices to 1, and sets the threshold to k = (m * r) / n
	
	PriceMechanism - All the papers start at price 1, and sets the threshold to k = (m * r) / n. 
	After a reviewer bids, the price of each paper becomes: min{1, r / d}
	
	MixedMechanism - The first iteration acts as FixedMechanism, the second iteration acts as PriceMechanism.
	
ReviewRequirements:

	positive integer
	
Behaviors:

	SincereIntegralBehavior - In an integral behavior each reviewer has 2 choices for bidding: {0, 1}. 
	Reviewers will submit a sincere integral bid with the lowest underbidding.
	
	BestTwoPreferenceBehavior - Each reviewer will bid 1 on all the papers that are in one of their 
	best two preferences rank.
	
	BestIntegralSincereUnderbidResponse - In an integral behavior each reviewer has 2 choices for bidding: 
	{0, 1}. The reviewer will submit a sincere underbid that will yield the minimal cost value according 
	to the private prices of the reviewer after allocation. (note: this behavior may have a long run time
	on a problem with a lot of papers and reviewers)
	
	BestIntegralSincereResponse - In an integral behavior each reviewer has 2 choices for bidding: 
	{0, 1}. The reviewer will submit a sincere bid that will yield the minimal cost value according
	to the private prices of the reviewer after allocation. (note: this behavior may have a long run time
	on a problem with a lot of papers and reviewers)
	

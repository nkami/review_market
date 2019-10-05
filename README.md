# A Market-Based Bidding Scheme for Peer Review Assignment
A simulation environment used for testing reviewers behavior in different bidding systems. The system is partitioned into three different parts:

1. [Cost Matrix and Quota Matrix Generation](#cost-matrix-and-quota-matrix-generation)
2. [Paper Bidding Simulation](#paper-bidding-simulation)
3. [Evaluation](#evaluation)

## Prerequisites
```
python 3.6
numpy 1.14.2
pandas 0.23.4
matplotlib 3.0.1
pathlib 2.32
pickle 4.0
gurobi 7+
```
It may be able to run with lower versions.

A free academic license and instructions on how to install gurobi are available here: http://www.gurobi.com/

## Cost Matrix and Quota Matrix Generation
In this part a json file with the following format is given as input:
```json
{
    	"papers_requirements": "fill",
	"private_prices_generator": "fill",
	"unallocated_papers_price": "fill",
	"instance_generator": "fill",
	"ignore_quota_constraints": "fill",
	"additional_params": "fill"
}
```
**papers requirements:** An integer that specifies the amount of reviews each paper requires.

**private prices generator:** Specifies the method of generating the private prices (utilities) of each paper for every reviewer. Currently the possible private prices generators are:
- SimplePrivatePricesGenerator - Assign a random private cost to each paper, while satisfying the condition that every paper with preference rank i > j will have lower cost than every paper with preference rank j (i.e. papers that are more desirable according to preference will have a lower cost).

**unallocated papers price:** A float that specifies the cost of an unallocated paper.

**instance generator:** Specifies the method of obtaining a preference profile of the reviewers. Currently the possible instance generators are:
- PreflibInstanceGenerator - the preference profile is obtained from a PrefLib file. (additional params: "PreflibFile": file path)

**ignore quota constraints:** A 'true' or 'false' boolean that indicates if the quota matrix will be used. If it's 'false' all of the elements of the matrix will be set to infinty, otherwise all elements will be set to 1 or 0 (depends if there is COI).

**additional params:** A dictionary that consists all of the additional parameters required for the previous chosen arguments. For example, for an instance generator of type 'PreflibInstanceGenerator', the key "PreflibFile" should be added with a value which is a file path.

#### command line -
```sh
python instance_generator.py $input_json_file_path
```
The output will be two json files: a cost matrix and a quota matrix created in the '.\output\cost_matrics' folder. The file names will be 'cost_matrix_{time_stamp}.json' and 'quota_matrix_{time_stamp}'. The cost matrix format is: 
```json
{
    "preflib_file": "path_to_file_used",
    "total_reviewers": "total_reviewers",
    "total_papers": "total_papers",
    "cost_matrix": "matrix"
}
```
The quota matrix format is: 
```json
{
    "preflib_file": "path_to_file_used",
    "total_reviewers": "total_reviewers",
    "total_papers": "total_papers",
    "quota_matrix": "matrix"
}
```
These files will be used as inputs to the second part.
#### example -
For a valid input json file:
```json
{
	"papers_requirements": 3,
	"private_prices_generator": "SimplePrivatePricesGenerator",
	"unallocated_papers_price": 1,
	"instance_generator": "PreflibInstanceGenerator",
	"ignore_quota_constraints": false,
	"additional_params": {"PreflibFile": ".//data//Preflib//bids_m3_n3.txt"}
}
```
Its output would be the following json files:
```json
{
    "preflib_file": ".//data//Preflib//bids_m3_n3.txt",
    "total_reviewers": 3,
    "total_papers": 3,
    "cost_matrix": [
        [
            0.7323764030043145,
            2.933268485530754,
            2.7386974509295876
        ],
        [
            2.7089642303259165,
            0.6090831838556042,
            0.6192301172499806
        ],
        [
            2.0542985547673065,
            0.8603096486030297,
            0.7168805606030314
        ]
    ]
}
```

```json
{
    "preflib_file": ".//data//Preflib//bids_m3_n3.txt",
    "total_reviewers": 3,
    "total_papers": 3,
    "quota_matrix": [
        [
            1.0,
            1.0,
            1.0
        ],
        [
            1.0,
            1.0,
            1.0
        ],
        [
            1.0,
            1.0,
            1.0
        ]
    ]
}
```
## Paper Bidding Simulation
The Paper Bidding Simulation part simulates the bidding process on the papers. The reviewers bid in a random order, and the papers' prices update according to the chosen arguments. In this part, a json file or a directory containing json files with the following format is given as input:

*Note: during simulation all combinations of parameters will be tested. In the example below there are 4 * 2  combinations, and for each combination 5 samples will be created.* 

```json
{
    "reviewers_behavior": "IntegralGreedyBidder",
    "fallback_behavior": "UniformBidder",
    "fallback_probability": [10, 20, 30, 75],
    "forced_permutations": [0],
    "number_of_bids_until_prices_update": [1],
    "total_bids_until_closure": [32],
    "matching_algorithm": ["Egalitarian", "FractionalAllocation"],
    "market_mechanism": "PriceMechanism",
    "ignore_quota_constraints": false,
    "additional_params": "fill",
    "price_weight": [2],
    "bidding_requirements": [6],
    "fallback_bidding_requirement": 10,
    "bidding_limit": 1000,
    "samples": 5,
    "cost_threshold": 8,
    "cost_threshold2": 0,
    "amount_of_csv_sample_outputs_per_100_samples": 1,
    "output_detail_level_permutations": 100,
    "output_detail_level_iterations": 2,
    "papers_requirements": 3,
    "unallocated_papers_price": 1,
    "random_matrices": true,
    "cost_matrix_path": ".//cost_matrices//cost_matrix_m52_n24.json",
    "quota_matrix_path": ".//cost_matrices//quota_matrix_m52_n24.json"
}
```
**reviewers behavior:** The chosen behavior of the reviewers during the paper bidding simulation. Currently the possible behaviors are:
- IntegralSelectiveBidder - on initialization, randomizes a price threshold for each paper as follows: papers with high cost have high chance to have a low or 0 threshold.
- UniformSelectiveBidder - behaves in a similar manner like IntegralSelectiveBidder but all the prices are uniform.
- IntegralGreedyBidder - orders papers according to (cost - price*weight) in increasing order. Bids until contribution exceeds the threshold. Only adds bids.
- UniformBidder - An integral behavior in which each reviewer has 2 choices for bidding: {0, 1}. Reviewers will submit a sincere integral bid until reaching the threshold.
- IntegralSincereBidder - Acts like IntegralGreedyBidder with weight = 0.

**fallback_probability:** The probability a reviewer will behave as chosen.

**fallback_behavior:** The default reviewer behavior.

**forced permutations:** An integer that specifies the number of times all the reviewers will place a bid before the papers' prices update (e.g., forced permutations that equals 5 would ensure that there are 5 rounds of bidding, in each round the reviewers bid once, and the papers' prices update after each round).

**number of bids until prices update:** An integer that specifies after how many submitted bids the papers' prices update (starting after the forced permuations finish).

**total bids until closure:** An integer that specifies the amount of bids until the auction closes (starting after the forced permuations finish).

**matching algorithm:** The chosen algorithm that will allocate the papers to the bidders at the end of the auction. Currently the possible algorithms are:
- FractionalAllocation - full description can be found in the paper.
- Utilitarian - run with the Max Utilitarian Objective.
- Egalitarian - run with the Egalitarian Objective.
- RankMaximal - run with the Rank Maximal Objective.
- LinearSumOWA - run with SUM-OAW with Linear OWA Objective.
- Nash - run with Max Nash Product Objective.

More information about the algorithms can be found [here](https://arxiv.org/abs/1705.06840).

**market mechanism:** The chosen market mechanism which controls the required threshold and how the papers' prices update. Currently the possible market mechanisms are:
- PriceMechanism - sets the starting price of all the papers to 1 and the threshold to: *sum(papers_requirements) / total_reviewers*. Each paper price is updated in the following way: *min{1, r_i / d_i}* where r_i and d_i are the amount of reviews required, and demand of paper i, respectively.

**price_weight:** Used for IntegralGreedyBidder.

**bidding_requirements:** The bidding requirements are used for the bidding mechanism, whereas the papers_requirements are used by the allocation algorithm. They do NOT have to agree in the general case with k.

**fallback_bidding_requirement:** This defines the default behavior for bidding_requirements.

**bidding_limit:** The limit of total bids a reviewer can make.

**samples:** Amount of samples simulated per combination of parameters.

**cost_threshold:** Threshold for first level bid.

**cost_threshold2:** Threshold for second level bid.

**amount_of_csv_sample_outputs_per_100_samples:** The number of sample csv files created (0 for none, 100 for all).

**output_detail_level_permutations:** Percent of permutation updates that will be printed.

**output_detail_level_iterations:** Percent of iteration updates that will be printed.

**random_matrices:** Use a random cost matrix from the "cost_matrices" directory (and its matching quota matrix) for each sample.

**cost_matrix_path:** A path to a json file containing a cost matrix (used if random_matrices is false).

**quota_matrix_path:** A path to a json file containing a quota matrix (used if random_matrices is false).

**additional params:** A dictionary that consists of all the additional parameters required for the previous chosen arguments.

#### command line -
```sh
python simulation.py $input_path
```
The output will be a new directory named "simulation_{time_stamp}" that will contain all of the simulations results.

![simulation output](https://github.com/nkami/review_market/blob/master/images/output_simulation_example.PNG)

## Evaluation

The evaluation part plots various metrics of the output of the simulation. A csv file of all the samples is expected (e.g., "all_sample.csv")
#### command line -
```sh
python evaluation.py $input_path
```

Currently 2 types of graphs are supported.

#### Bids vs. Allocation
![bids vs allocation](https://github.com/nkami/review_market/blob/master/images/allocation%20vs%20bids.png)

#### Allocation Fairness
![allocation fairness](https://github.com/nkami/review_market/blob/master/images/allocation%20fairness.png)







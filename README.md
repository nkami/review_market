# A Market-Based Bidding Scheme for Peer Review Assignment
A simulation environment used for testing reviewers behavior in different bidding systems. The system is partitioned into two different parts:

1. [Cost Matrix and Quota Matrix Generation](#Cost_Matrix_and_Quota_Matrix_Generation)
2. [Paper Bidding Simulation](#Paper_Bidding_Simulation)

## Prerequisites
```
python 3.6
numpy 1.14.2
pandas 0.23.4
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
**papers requirements:** an integer that specifies the amount of reviews each paper requires.
**private prices generator:** specifies the method of generating the private prices (utilities) of each paper for every reviewer. Currently the possible private prices generators are:
- SimplePrivatePricesGenerator - Assign a random private cost to each paper, while satisfying the condition that every paper with preference rank i > j will have lower cost than every paper with preference rank j (i.e. papers that are more desirable according to preference will have a lower cost).

**unallocated papers price:** a float that specifies the cost of an unallocated paper.
**instance generator:** specifies the method of obtaining a preference profile of the reviewers. Currently the possible instance generators are:
- PreflibInstanceGenerator - the preference profile is obtained from a PrefLib file. (additional params: "PreflibFile": file path)

**ignore quota constraints:** a 'true' or 'false' boolean that indicates if the quota matrix will be used. If it's 'false' all the elements of the matrix will be set to infinty, otherwise all elements will be set to 1 or 0 (depends if there is COI).

**additional params:** a dictionary that consists all the additional parameters required for the previous chosen arguments. For example, for and instance generator of type 'PreflibInstanceGenerator', the key "PreflibFile" with a file path as it's value should be added.

#### command line -
```sh
python instance_generator.py $input_json_file_path
```
The output will be two json files: a cost matrix and a quota matrix created in '.\output\cost_matrics' folder. The files names will be 'cost_matrix_{time_stamp}.json', 'quota_matrix_{time_stamp}'. the cost matrix format is: 
```json
{
    "preflib_file": "path_to_file_used",
    "total_reviewers": "total_reviewers",
    "total_papers": "total_papers",
    "cost_matrix": "matrix"
}
```
the quota matrix format is: 
```json
{
    "preflib_file": "path_to_file_used",
    "total_reviewers": "total_reviewers",
    "total_papers": "total_papers",
    "quota_matrix": "matrix"
}
```
These files will be used as an input for the second part.
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
It's output would be the following json files:
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
The Paper Bidding Simulation part simulates the bidding process on the papers. The reviewers bid in a random order, and the papers prices update according to the chosen arguments. In this part a json file or a directory containing json files with the following format is given as input:

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
**reviewers behavior:** the chosen behavior of the reviewers during the paper bidding simulation. Currently the possible behaviors are:
- IntegralSelectiveBidder - on initialization randomizes a price threshold for each paper as follows: papers with high cost have high chance to have a low or 0 threshold.
- UniformSelectiveBidder - behaves in a similar manner like IntegralSelectiveBidder but all the prices are uniform.
- IntegralGreedyBidder - Orders papers according to (cost - price*weight) in increasing order. Bids until contribution exceeds the threshold. Only adds bids.
- UniformBidder - An integral behavior in which each reviewer has 2 choices for bidding: {0, 1}. Reviewers will submit a sincere integral bid until reaching the threshold.
- IntegralSincereBidder - Acts like IntegralGreedyBidder with weight = 0.

**fallback_probability:** The probability a reviewer will behave as chosen.

**fallback_behavior:** The default reviewer behavior.

**forced permutations:** an integer that specifies the number of times all the reviewers will place a bid before the papers prices update (e.g. forced permutations that is 5 would ensure that all the reviewers bid 5 times and the papers prices would update after each round of bids).

**number of bids until prices update:** an integer that specifies after how many submitted bids the papers price updates (excluding the forced permuations argument).

**total bids until closure:** an integer that specifies the amount of bids until the auction closes (excluding the forced permuations argument).

**matching algorithm:** the chosen algorithm that will allocate the papers to the bidders at the end of the auction. In this part the allocation does not take place, but some reviewers behaviors require the algorithm (for anticipating their best bid). Currently the possible algorithms are:
- FractionalAllocation - full description can be found in the paper.
- Utilitarian - run with the Max Utilitarian Objective.
- Egalitarian - run with the Egalitarian Objective.
- RankMaximal - run with the Rank Maximal Objective.
- LinearSumOWA - run with SUM-OAW with Linear OWA Objective.
- Nash - run with Max Nash Product Objective.

More information about the algorithms can be found [here](https://arxiv.org/abs/1705.06840).

**market mechanism:** the chosen market mechanism which controls the required threshold and how the papers prices update. Currently the possible market mechanisms are:
- PriceMechanism - sets the starting price of all the papers to 1 and the threshold to: *sum(papers_requirements) / total_reviewers*. Each paper price is updated in the following way: *min{1, r_i / d_i}* where r_i and d_i are the amount of reviews required and demand of paper i respectively.

**additional params:** A dictionary that consists all the additional parameters required for the previous chosen arguments.

**price_weight:** Used for IntegralGreedyBidder.

**bidding_requirements:** The bidding requirement is used for the bidding mechanism, whereas the papers_requirement is used by the allocation algorithm. They do NOT have to agree in the general case with k.

**fallback_bidding_requirement:** bidding_requirements for default behavior.

**bidding_limit:** The limit of total bids a reviewer can make.

**samples:** Amount of samples simulated per combination of parameters.

**cost_threshold:** Threshold for first level bid.

**cost_threshold2:** Threshold for second level bid.

**amount_of_csv_sample_outputs_per_100_samples:** The number of samples csv files created (0 for none, 100 for all).

**output_detail_level_permutations:** Percent of permutation updates that will be printed.

**output_detail_level_iterations:** Percent of iteration updates that will be printed.

**random_matrices:** Use a random cost matrix from the "cost_matrices" directory (and its matching quota matrix) for each sample.

**cost_matrix_path:** A path to a json file containing a cost matrix (used if random_matrices is false).

**quota_matrix_path:** A path to a json file containing a quota matrix (used if random_matrices is false).

#### command line -
```sh
python simulation.py $input_path
```
The output will be a new directory named as "simulation_{time_stamp}" that will contain all the simulations results.


![Alt text](./images/output_simulation_example.png?raw=true "Title")

**#step:** the bid number (i.e #step 0 will be the first bid).

**reviewer:** the reviewer id of the reviewer who made the bid.

**updates:** the number of times the papers price has been updated.

**paper id:** the paper id of the paper being bid on.

**private cost:** the private cost (utility) of the paper to the reviewer.

**price:** the price of the paper when the bid was made.

**bid:** the amount the reviewer bidded on the paper.

**total bid:** the sum of all the bids of a reviewer.

**total price:** dot product of all the bids and papers prices of a reviewer - <p, b>.

**total private cost:** dot product of all the bids and papers private prices of a reviewer - <p_private, b>.

**matching output json file:** the name of the first output json file.

# sep
![Alt text](./images/output_allocation_example.png?raw=true "Title")

**reviewer:** the reviewer id of the reviewer who made the bid. An id of -1 represents a dummy reviewer that gets all the unallocated papers after the allocation.

**paper id:** the paper id of the relevant paper.

**private cost:** the private cost (utility) of the paper to the reviewer.

**step 1 allocation:** the allocated part of the paper to the reviewer after the first step of the FractionalAllocation algorithm. A value of -1 is given if its irrelevant (e.g. the used algorithm is not FractionalAllocation, or its the dummy reviewer of unallocated papers).

**step 2 allocation:** the allocated part of the paper to the reviewer after the second step of the FractionalAllocation algorithm. A value of -1 is given if its irrelevant (e.g. the used algorithm is not FractionalAllocation, or its the dummy reviewer of unallocated papers).

**step 3 allocation:** the allocated part of the paper to the reviewer after the third and final step of the FractionalAllocation algorithm. The unallocated part of a paper and other algorithms allocations will be placed here as well.

**cost contribution:** the product of the private cost and the bid of the relevant paper.

**individual contribution:** the sum of the cost contribution of the relevant reviewer.

**total social cost:** the sum of all individual contributions.

**total unallocated papers:** the total amount of papers left unallocated after using the allocation algorithm.

**total paper cost:** the sum of all the products of the bid and private price for all reviewers of the relevant paper.

**matching input json file:** the name of the input json file used.





# Market Based Assignments for Peer Review
A simulation environment used for testing reviewers behavior in different bidding systems. The system is partitioned into three different parts:

1. [Cost Matrix and Quota Matrix Generation](#cost-matrix-and-quota-matrix-generation)
2. [Paper Bidding Simulation](#paper-bidding-simulation)
3. [Allocation and Final Evaluations](#allocation-and-final-evaluations)

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
The output will be a json file created in a new (or old if it already exists) directory: 'output'. The file name will be 'cost_matrix_{time_stamp}.json', and it's format:
```json
{
    "reviewers_behavior": "fill",
    "forced_permutations": "fill",
    "number_of_bids_until_prices_update": "fill",
    "total_bids_until_closure": "fill",
    "matching_algorithm": "fill",
    "market_mechanism": "fill",
    "ignore_quota_constraints": "derived from the input json file",
    "additional_params": "fill",
    "total_reviewers": "derived from the input json file",
    "total_papers": "derived from the input json file",
    "papers_requirements": "derived from the input json file",
    "unallocated_papers_price": "derived from the input json file",
    "cost_matrix": "derived from the input json file",
    "quota_matrix": "derived from the input json file"
}
```
This file will be used as an input for the second part.
#### example -
For a valid input json file:
```json
{
	"papers_requirements": 2,
	"private_prices_generator": "SimplePrivatePricesGenerator",
	"unallocated_papers_price": 1,
	"instance_generator": "PreflibInstanceGenerator",
	"ignore_quota_constraints": false,
	"additional_params": {"PreflibFile": ".//data//Preflib//example_quota.txt"}
}
```
It's output would be the following json file:
```json
{
    "reviewers_behavior": "fill",
    "forced_permutations": "fill",
    "number_of_bids_until_prices_update": "fill",
    "total_bids_until_closure": "fill",
    "matching_algorithm": "fill",
    "market_mechanism": "fill",
    "ignore_quota_constraints": false,
    "additional_params": "fill",
    "total_reviewers": 3,
    "total_papers": 3,
    "papers_requirements": [
        2,
        2,
        2
    ],
    "unallocated_papers_price": [
        1,
        1,
        1
    ],
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
    ],
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
The Paper Bidding Simulation part simulates the bidding process on the papers. The reviewers bid in a random order, and the papers prices update according to the chosen arguments. In this part a json file with the following format is given as input:
```json
{
    "reviewers_behavior": "fill",
    "forced_permutations": "fill",
    "number_of_bids_until_prices_update": "fill",
    "total_bids_until_closure": "fill",
    "matching_algorithm": "fill",
    "market_mechanism": "fill",
    "ignore_quota_constraints": "derived from the input json file",
    "additional_params": "fill",
    "total_reviewers": "derived from the input json file",
    "total_papers": "derived from the input json file",
    "papers_requirements": "derived from the input json file",
    "unallocated_papers_price": "derived from the input json file",
    "cost_matrix": "derived from the input json file",
    "quota_matrix": "derived from the input json file"
}
```
**reviewers behavior:** the chosen behavior of the reviewers during the paper bidding simulation. Currently the possible behaviors are:
- SincereIntegralBehavior - In an integral behavior each reviewer can bid 0 or 1. Reviewers will submit a sincere integral bid with the lowest underbidding.
- BestIntegralSincereUnderbidResponse - In an integral behavior each reviewer can bid 0 or 1. The reviewer will submit a sincere underbid that will yield the minimal cost value according to the private prices of the reviewer after allocation.
- BestIntegralSincereResponse - In an integral behavior each reviewer can bid 0 or 1. The reviewer will submit a sincere bid that will yield the minimal cost value according to the private prices of the reviewer after allocation.

**forced permutations:** an integer that specifies the number of times all the reviewers will place a bid before the papers prices update (e.g. forced permutations that is 5 would ensure that all the reviewers bid 5 times and the papers prices would update after each round of bids).

**number of bids until prices update:** an integer that specifies after how many submitted bids the papers price updates (excluding the forced permuations argument).

**total bids until closure:** an integer that specifies the amount of bids until the auction closes (excluding the forced permuations argument).

**matching algorithm:** the chosen algorithm that will allocate the papers to the bidders at the end of the auction. In this part the allocation does not take place, but some reviewers behaviors require the algorithm (for anticipating their best bid). Currently the possible algorithms are:
- FractionalAllocation - full description can be found in the paper.

**market mechanism:** the chosen market mechanism which controls the required threshold and how the papers prices update. Currently the possible market mechanisms are:
- PriceMechanism - sets the starting price of all the papers to 1 and the threshold to: *sum(papers_requirements) / total_reviewers*. Each paper price is updated in the following way: *min{1, r_i / d_i}* where r_i and d_i are the amount of reviews required and demand of paper i respectively.

**additional params:** a dictionary that consists all the additional parameters required for the previous chosen arguments.

#### command line -
```sh
python prices_mechanisms.py $input_json_file_path
```
The output will be 2 files with matching time stampts. The first one is a json file created in a new (or old if it already exists) directory: 'output'. The file name will be 'simulation_{time_stamp}.json', and it's format:
```json
{
    "reviewers_behavior": "derived from the input json file",
    "forced_permutations": "derived from the input json file",
    "number_of_bids_until_prices_update": "derived from the input json file",
    "total_bids_until_closure": "derived from the input json file",
    "matching_algorithm": "derived from the input json file",
    "market_mechanism": "derived from the input json file",
    "ignore_quota_constraints": "derived from the input json file",
    "additional_params": "derived from the input json file",
    "total_reviewers": "derived from the input json file",
    "total_papers": "derived from the input json file",
    "final_bidding_profile": "derived from the input json file",
    "papers_requirements": "derived from the input json file",
    "unallocated_papers_price": "derived from the input json file",
    "cost_matrix": "derived from the input json file",
    "quota_matrix": "derived from the input json file"
}
```
This file will be used as input for the third part. The second output file is a csv file created in a new (or old if it already exists) directory: 'output'. The file name will be 'simulation_{time_stamp}.csv', and it's format:

![alt text](https://github.com/nkami/review_market/blob/master/images/output_simulation_example.PNG)

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

#### example -
For a valid input json file:
```json
{
    "reviewers_behavior": "BestIntegralSincereResponse",
    "forced_permutations": 1,
    "number_of_bids_until_prices_update": 12,
    "total_bids_until_closure": 30,
    "matching_algorithm": "FractionalAllocation",
    "market_mechanism": "PriceMechanism",
    "ignore_quota_constraints": false,
    "additional_params": {},
    "total_reviewers": 3,
    "total_papers": 3,
    "papers_requirements": [
        2,
        2,
        2
    ],
    "unallocated_papers_price": [
        1,
        1,
        1
    ],
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
    ],
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

## Allocation and Final Evaluations
In this part the final allocation takes place, and various metrics are being measured. A json file with the following format should be used as input:
```json
{
    "reviewers_behavior": "derived from the input json file",
    "forced_permutations": "derived from the input json file",
    "number_of_bids_until_prices_update": "derived from the input json file",
    "total_bids_until_closure": "derived from the input json file",
    "matching_algorithm": "derived from the input json file",
    "market_mechanism": "derived from the input json file",
    "ignore_quota_constraints": "derived from the input json file",
    "additional_params": "derived from the input json file",
    "total_reviewers": "derived from the input json file",
    "total_papers": "derived from the input json file",
    "final_bidding_profile": "derived from the input json file",
    "papers_requirements": "derived from the input json file",
    "unallocated_papers_price": "derived from the input json file",
    "cost_matrix": "derived from the input json file",
    "quota_matrix": "derived from the input json file"
}
```
**final bidding profile:** the final bids on all the papers after the auction closed.

#### command line -
```sh
python Match.py $input_json_file_path
```
 The output file is a csv file created in a new (or old if it already exists) directory: 'output'. The file name will be 'final_allocation_{time_stamp}.csv', and it's format:
 
![alt text](https://github.com/nkami/review_market/blob/master/images/output_allocation_example.PNG)

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





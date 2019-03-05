 
MD-00002: Computer Science Conference Bidding Data
This dataset contains the bidding data from 3 Computer Science Conferences.
 This contains the bids of all reviewers (aside a small number of opt-outs) over a subset of papers at the conference.

The bidding language for these conferences is yes/maybe/conflict.
 In order to make these more useful for PreLib users,
 we have converted them to incomplete partial orders of the form {yes} > {maybe} > {no response}.
 The papers for which a reviewer had a conflict have been removed from their preference list.
 All reviewers had different preference orderings, hence each file contains as many entries as reviewers.
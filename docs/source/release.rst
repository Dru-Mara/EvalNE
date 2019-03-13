Release Log
===========

EvalNE v0.2.1
-------------

Release date: 13 Mar 2019

New features
~~~~~~~~~~~~
- Added `WRITE_WEIGHTS_OTHER` in conf files which allows the user to specify if the input train network to the NE methods should have weights or not. If True but the original input network is unweighted, weights of 1 are given to each edge. This features is useful for e.g. the original code of LINE, which requires edges to have weights (all 1 if the graph is unweighted).
- Added `WRITE_DIR_OTHER` in conf files which allows the user to specify if the input train network to the NE methods should be specified with both directions of edges or a single one.
- Added `SEED` in the conf file which sets a general random seed for the whole library. If None the system time is used.
- Added a faster method for splitting non-edges in train and test when all non-edges in the graph are required.

Documentation
~~~~~~~~~~~~~
- Readme and docs update
- Descriptions of each option in conf.ini added

Miscelaneous
~~~~~~~~~~~~
- Removed optional seed parameter from all methods in split_train_test.py
- Removed random seed resetting in the edges split methods
- `simple-example.py` now checks if OpenNE is installed, if not it runs only the LP heuristics.
- Sklearn removed from requirements.txt (already satisfied by scikit-learn)
- `setup.py` update. Ready for making EvalNE pip installable.
- Train/validation fraction was 50/50 which caused the train set to be excesively small and parameter validation not accurate. New value is 90/10.
- Improved warnings in evaluator code
- General code cleaning

Bugs
~~~~
- train/validation and train/test splits used the same random seed for generating the edge split which caused correlation between them. Now the train/validation split is random. 
- Fixed a bug which would cause the evaluation of any edge embedding method to crash.
- Precitions from edge embeddings were computed using LogisticRegression.predict(). This gives class labels and not class probabilities resulting in worst estimates of method performance. This has been changed to LogisticRegression.predict_proba()


EvalNE v0.2.0
-------------

Release date: 4 Feb 2019

API changes
~~~~~~~~~~~
- The evaluate_ne_cmd method has been renamed to evaluate_cmd
- evaluate_cmd can now evaluate node, edge or end to end embedding method
- evaluate_cmd a new method_type parameter has been added to indicate how the method should be evaluated (ne, ee or e2e)
- ScoreSheet object has been removed
- Score method removed from Katz and KatzApprox classes
- Method get_parameters() from Evaluator has been removed

New features
~~~~~~~~~~~~
- Added method_type option in *.ini* files to evaluate (ne, ee or e2e)
- compute_results method now takes an optional label binarizer parameter
- evaluate_ne method now takes an optional label binarizer parameter
- save and pretty_print methods in Results now take a precatk_vals parameter which indcates for which k values to compute this score
- When REPORT SCORES = all is selected in the *.ini* file, the library now presents all the available metrics for each algorithm and dataset averaged over the number of repetitions.

Documentation
~~~~~~~~~~~~~
- Docstring updates
- Release log added to Docs
- Contributing added to Docs

Miscelaneous
~~~~~~~~~~~~
- Exception handling improvements

Bugs
~~~~
- Prevented possible infinite loop while generating non-edges by raising a warning if the used-selected values is > that the max possible non-edges.






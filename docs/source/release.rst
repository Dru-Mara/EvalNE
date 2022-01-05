Release Log
===========

EvalNE v0.3.4
-------------

Release date: 05 Dec 2022

Documentation
~~~~~~~~~~~~~
- Release log update.
- Docstring improvements affecting some classes and functions.
- Improved Readme.md and docs files.

New features
~~~~~~~~~~~~
- Included a bash script to control the number of threads used by numpy during execution.
- Included a new label binarization method ('prop') which binarizes predictions based on the number of positive/negative instances in the train data.
- The library now logs the logistic regression coefficients per method when LR or LRCV are used.
- Included a new performance metric, average precision for the LP, SP, and NR tasks.
- Parameter checks in the EvalSetup class for .ini configuration files can now be turned on or off. 
- New parallel coordinates plot has been added to visualize method performance from output pickle files. 

Miscellaneous
~~~~~~~~~~~~
- Input type errors are now cached and logged and no longer cause the evaluation to crash.


EvalNE v0.3.3
-------------

Release date: 14 Dec 2020

Documentation
~~~~~~~~~~~~~
- Release log update.
- Extensive docstring improvements affecting all classes and functions.
- Docstring examples included for all important classes and low level functions.
- Improved variable descriptions in conf.ini.
- Improved Readme.md and docs files.

New features
~~~~~~~~~~~~
- Sign prediction added as a downstream task that can be evaluated (using the SPEvaluator class).
- Three new classes (LPEvalSplit, SPEvalSplit and NREvalSplit) added that simplify the computation of evaluation splits for the LP, SP and NR downstream tasks.
- Added three new heuristic baselines: Cosine Similarity, Leicht-Holme-Newman index and Topological Overlap.
- When used as a command line tool, the library now provides both the train and test evaluation scores in the output folder.
- When used as an API the user can now conveniently store the model predictions for any downstream task.
- Added timeout for baselines evaluation
- Added function that can run other functions in a separate process with given timeout.

Miscellaneous
~~~~~~~~~~~~
- Improved requirement specification in requirements.txt and setup.py.
- Improved library and module level imports.
- General improvements on error and warning messages.
- Memory errors are now catched and logged.
- All numerical output is now rounded to 4 decimals.

Bugs
~~~~
- Fixed a bug that would cause a TimeoutError to be raised incorrectly.


EvalNE v0.3.2
-------------

Release date: 10 Dec 2019

Documentation
~~~~~~~~~~~~~
- Release log update
- Various docstring improvements
- Improved variable descriptions in conf.ini

New features
~~~~~~~~~~~~
- The user can now set a timeout for the execution of each method in the conf files. E.g. TIMEOUT = 1800
- Conf files now support any sklearn binary classifer in the LP_MODEL variable. E.g. LP_MODEL=sklearn.svm.LinearSVC(C=1.0, kernel=’rbf’, degree=3)
- Conf files also support keyword SVM for the LP_MODEL. This uses the sklearn LinearSVC model and tunes the regularization parameter on a grid [0.1, 1, 10, 100, 1000].
- Method execution is made safer by using Popen communicate instead of subprocess.run(shell=True)
- Removed lp_model coefficient output. This could lead to errors and failed evaluations for certain Sklearn binary classifiers
- Method compute_pred() of LPEvaluator and NREvaluator classes now tries to use lp_model.predict_proba() if the classifier does not have it, the function defaults to lp_model.predict()
- The scoresheet method get_pandas_df() now includes a repeat parameter which denotes the exact experiment repeat results the user wants in the DF. If repeat=None, the DF returned will contain the average metric over all experiment repeats. 

Miscellaneous
~~~~~~~~~~~~
- Log file output now shows timeout errors and LR method selected
- Corrected the cases where some warnings were reported as errors
- Added util.py in the utils module

Bugs
~~~~
- Fixed bug which would prevent the library to store the output when executed from Py3


EvalNE v0.3.1
-------------

Release date: 2 Nov 2019

Documentation
~~~~~~~~~~~~~
- Release log update
- Various docstring improvements

New features
~~~~~~~~~~~~
- New heuristic for LP named `all_baselines`. Generates a 5-dim edge embedding by combining the existing heuristics [CN, JC, AA, PA, RAI].
- Automated file headder detection (in the output of embedding methods) is now a function
- Functions for reading the embeddings, predictions and node labels have been added
 

Miscellaneous
~~~~~~~~~~~~
- General improvements in NC task
- Added NCScores and NCResults classes
- Pickle file containig evaluation results is now saved incrementally, after each networks has been evaluated. If the user stops the process mid-way the results up to the last network will be available 
- Coefficients of the binary classifier per evaluated method are now provided for LP and NR tasks
- Improved exception management
- Improved conf file sanity checks
- Evaluated methods now return a single Results object instead of a list 

Bugs
~~~~
- Fixed bug related to plotting PR and AUC curves
- Fixed node classification bugs preventing the evaluaition to run properly


EvalNE v0.3.0
-------------

Release date: 21 Oct 2019

Documentation
~~~~~~~~~~~~~
- Release log update

New features
~~~~~~~~~~~~
- Old Evaluator class is now LPEvaluator
- Added Network Reconstruction evaluation (NREvaluator)
- Added Node Classification evaluation (NCEvaluator)
- Train/validation splits are now required when initializing Evaluator classes
- Added 3 new algorithms for computing train/test splits. One extremely scalable up to millions of nodes/edges
- Improved error management and error logging
- Edge embedding methods are now always tunned as method parameters. Results for the best are given.
- For link prediction and network reconstruction the user can now evaluate the methods exclusively on train data.
- Addes Scoresheet class to simplify output management
- Export results directly to pandas dataframe and latex tables suppored

Miscellaneous
~~~~~~~~~~~~
- Changed default parameters for EvalSplit
- Added new parameter for EvalSplit.set_split()
- Evaluation output is now always stored as pickle file
- Execution time per method and dataset is not provided
- Train/test average time per dataset is registered
- Added `auto` mode for the Results class to decide if train or test data should be logged


EvalNE v0.2.3
-------------

Release date: 25 Apr 2019

Documentation
~~~~~~~~~~~~~
- Release log update
- Library diagram minor update

Bugs
~~~~
- Corrected parameter tuning rutine which was minimizing the objective metric given instead of maximizing it.
- Corrected evaluate_cmd() function output.

New features
~~~~~~~~~~~~
- Evaluation output file now contains also a table of execution times per evaluated method.

Miscellaneous
~~~~~~~~~~~~
- Changed behaviour of verbosity flag. Now, if Verbose=False it deactivates all stdout for the methods being evaluated (not stderr) but maintains the library stdout.
- Added more conf.ini files for reproducing the experimental section of different papers.


EvalNE v0.2.2
-------------

Release date: 14 Mar 2019

Documentation
~~~~~~~~~~~~~
- Readme and docs update to include pip installation

Miscelaneous
~~~~~~~~~~~~
- Library is now pip installable
- Minor bugfixes


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

Miscellaneous
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

Miscellaneous
~~~~~~~~~~~~
- Exception handling improvements

Bugs
~~~~
- Prevented possible infinite loop while generating non-edges by raising a warning if the used-selected values is > that the max possible non-edges.






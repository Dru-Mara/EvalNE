Release Log
===========

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






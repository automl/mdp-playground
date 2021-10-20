Welcome to MDPP's documentation!
==================================

You can click through the tree on the left for MDPP's API reference. See below for how to call MDPP from the command line.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   mdp_playground

CLI Reference
***********************

.. argparse::
  :module: mdp_playground.scripts.run_experiments
  :func: generate_parser
  :prog: run-mdpp-experiments

# af2_conformations_ds
My version of the scripts to run AF2 with shallower MSA. To run with templates users need to modify the new__ files generated in the running folder, kill the run and copy the modified files in the *_env folder.
This ugly approach allows users to manually select the best 4 templates.

In the predict.py file there is a function called predict_structure_from_template that instead takes a PDB in input.

T1195_1632.py is an example on how to run locally without templates.
pth1r_1632_template.py is an example on how to run locally with templates (following the aforementioned procedure)

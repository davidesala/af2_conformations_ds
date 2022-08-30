from . import mmseqs2
from . import util
import os
import numpy as np
from typing import NoReturn
import random
import sys

import Bio.PDB

import alphafold
from alphafold.common import protein
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.model import modules
from alphafold.model import folding
from alphafold.data.tools import hhsearch

import jax.numpy as jnp
import jax

import pickle

def set_config(
    use_templates: bool,
    max_msa_clusters: int,
    max_extra_msa: int,
    max_recycles: int,
    model_id: int,
    n_struct_module_repeats: int,
    n_features_in: int,
    model_params: int=0
  ): # -> alphafold.model.RunModel:
  r""" Generated Runner object for AlphaFold

  Args:
    use_templates: Whether templates are used
    max_msa_cluster: How many extra sequences to use
    max_extra_msa: How many extra sequences to use? Unclear
    max_recycles: Number of recycling iterations to use
    tol: RMSD tolerance before prematurely exiting due to convergence
    model_id: Which AF2 model to use
    n_struct_module_repeats: Number of passes through structure module

  Output:
    AlphaFold config object
  """

  assert model_id in range( 1, 6 )

  if model_params not in range( 1, 6 ):
    model_params = model_id

  print( "Prediction parameters:" )
  print( f"\tModel ID: { model_id }" )
  print( f"\tUsing templates: { use_templates }" )
  print( f"\tMaximum number of MSA clusters: { max_msa_clusters }" )
  print( f"\tMaximum number of extra MSA clusters: { max_extra_msa }" )
  print( f"\tMaximum number of recycling iterations: { max_recycles }" )

  cfg = alphafold.model.config.model_config( f"model_{ model_params }_ptm" )

  # Here we provide config settings
  cfg.data.eval.num_ensemble = 1
  cfg.data.eval.max_msa_clusters = min(
      n_features_in, max_msa_clusters )
  cfg.data.common.max_extra_msa = max( 1, min(
      n_features_in - max_msa_clusters, max_extra_msa )
  )
  cfg.data.common.num_recycle = max_recycles
  cfg.model.num_recycle = max_recycles
  cfg.data.common.use_templates = use_templates
  cfg.model.embeddings_and_evoformer.template.embed_torsion_angles = \
    use_templates
  cfg.model.embeddings_and_evoformer.template.enabled = use_templates
  cfg.data.common.reduce_msa_clusters_by_max_templates = use_templates
  cfg.data.eval.subsample_templates = use_templates

  #cfg.data.eval.max_templates = n_template
  #cfg.model.embeddings_and_evoformer.template.max_template = n_template
  cfg.model.heads.structure_module.num_layer = n_struct_module_repeats

  params = alphafold.model.data.get_model_haiku_params(
      model_name=f"model_{ model_id }_ptm",
      data_dir="." )

  return alphafold.model.model.RunModel( cfg, params )

def predict_structure_old(
    job: str,
    seq: str,
    idx: int,
    a3m_lines,
    template_paths=None,
    use_templates=False,
    random_seed=None,
    max_msa_clusters=512,
    max_extra_msa=1024,
    tol=0,
    max_recycles=3,
    n_template=4,
    n_struct_module_repeats=8
  ):

  r""" Predicts the structure.

  Args:
  	job: Name of the job
  	seq: Sequence
  	use_templates: Whether to use templates or not
  	n_models: Number of models to generate (split between models 1 and 2)
  	idx: ID to give to filename
  	random_seed: Random seed
  	max_msa_clusters: Number of sequences
  	max_extra_msa: ???
  	tol: Tolerance for premature exit
  	max_recycles: Maximum number of iterations to run

  Output:
  	None
  """

  move_prefix = "new_"

  if random_seed is None:
    random_seed = random.randrange( sys.maxsize )

  model_id = random.randint( 1, 2 )

  if use_templates and template_paths is not None:
    if n_template > 0:
      util.randomize_templates( template_paths, move_prefix, n_template )
    tfeatures_in = util.setup_features(
        seq, a3m_lines, util.mk_template(seq, a3m_lines, template_paths).features)
#    tfeatures = util.mk_template( seq, a3m_lines, template_paths )
#    tfeatures_in = tfeatures.features
    #print( f"{ len( tfeatures_in.errors ) } errors with templates" )
    #print( f"{ len( tfeatures_in.warnings ) } warnings with templates" )
  else:
    del template_paths # It is None if use_templates is false
    tfeatures_in = util.mk_mock_template( seq )

  # Assemble the dictionary of input features
  #features_in = util.setup_features( seq, a3m_lines, tfeatures_in )

  # Run the models
  model_runner = set_config(
    use_templates,
    max_msa_clusters,
    max_extra_msa,
    max_recycles,
    model_id,
    n_struct_module_repeats,
    len( tfeatures_in[ "msa" ] )
  )

  features = model_runner.process_features(
      tfeatures_in,
      random_seed=random_seed
  )

  # Generate the model
  result = model_runner.predict( features )
  pred = alphafold.common.protein.from_prediction( features, result )

  outname = "_".join( map( str, (
      job,
      idx,
    ) ) ) + ".pdb"

  to_pdb(
      job,
      outname,
      pred,
      result[ 'plddt' ],
      features_in[ 'residue_index' ]
  )

  if n_template > 0:
    util.reset_pdb70( template_paths, move_prefix )

  return outname, result

def predict_structure_no_templates(
    job: str,
    seq: str,
    idx: int,
    a3m_lines,
    model_id=None,
    model_params=None,
    random_seed=None,
    max_msa_clusters=512,
    max_extra_msa=1024,
    max_recycles=3,
    n_struct_module_repeats=8
  ):

  r""" Predicts the structure.

  Args:
    job: Name of the job
    seq: Sequence
    n_models: Number of models to generate (split between models 1 and 2)
    idx: ID to give to filename
    random_seed: Random seed
    max_msa_clusters: Number of sequences
    max_extra_msa: ???
    max_recycles: Maximum number of iterations to run

  Output:
    None
  """

  if model_id not in range( 1, 6 ) or model_id is None:
    model_id = random.randint( 1, 5 )
  if model_params not in range( 1, 6 ) or model_id is None:
    model_params = model_id

  if random_seed is None:
    seed = random.randrange( sys.maxsize )
  else:
    seed = random_seed

  features_in = util.setup_features( seq, a3m_lines, util.mk_mock_template( seq ) )

  model_runner = set_config(
    False,
    max_msa_clusters,
    max_extra_msa,
    max_recycles,
    model_id,
    n_struct_module_repeats,
    len( features_in[ "msa" ] ),
    model_params=model_params
  )

  features = model_runner.process_features(
      features_in,
      random_seed=random_seed
    )

  #let's try to add here distogram
     #load caustom pickle
#  with open("mct1_edited.pkl", 'rb') as f:
#      print("Edited Distogram loaded")
#      tmp_distogram = pickle.load(f)
#  result = tmp_distogram	
  # Generate the model
  result = model_runner.predict( features )
  pred = alphafold.common.protein.from_prediction( features, result )

  outname = "_".join( map( str, (
      job,
      idx
    ) ) ) + ".pdb"

  to_pdb(
      job,
      outname,
      pred,
      result[ 'plddt' ],
      features_in[ 'residue_index' ]
  )

  # Important for multi-iteration runs

  return outname, result

def predict_structure_from_template(
    job: str,
    seq: str,
    idx: int,
    a3m_lines,
    template_pdb: str,
    n_iter=1,
    model_id=1,
    random_seed=None,
    max_msa_clusters=32,
    max_extra_msa=64,
    tol=0,
    max_recycles=3,
    n_struct_module_repeats=8
  ):

  r""" Predicts the structure.

  Args:
    job: Name of the job
    seq: Sequence
    n_models: Number of models to generate (split between models 1 and 2)
    idx: ID to give to filename
    random_seed: Random seed
    max_msa_clusters: Number of sequences
    max_extra_msa: ???
    tol: Tolerance for premature exit
    max_recycles: Maximum number of iterations to run

  Output:
    None
  """



  if model_id not in [ 1, 2 ]:
    model_id = random.randint( 1, 2 )

  for i in range( n_iter ):

    if random_seed is None:
      seed = random.randrange( sys.maxsize )
    else:
      seed = random_seed

    pdb = protein.from_pdb_string( util.pdb2str( template_pdb ) )

    tfeatures_in = {
      "template_aatype" : jax.nn.one_hot( pdb.aatype, 22 )[ : ][ None ],
      "template_all_atom_masks" : pdb.atom_mask[ : ][ None ],
      "template_all_atom_positions" : pdb.atom_positions[ :][ None ],
      "template_domain_names" : np.asarray( [ "None" ] ) }

    # Assemble the dictionary of input features
    features_in = util.setup_features( seq, a3m_lines, tfeatures_in )

    model_runner = set_config(
      True,
      max_msa_clusters,
      max_extra_msa,
      max_recycles,
      model_id,
      n_struct_module_repeats,
      len( features_in[ "msa" ] )
    )

    features = model_runner.process_features(
        features_in,
        random_seed=random_seed
    )

    # Generate the model
    result = model_runner.predict( features )
    pred = alphafold.common.protein.from_prediction( features, result )

    outname = "_".join( map( str, (
        job,
        template_pdb.split( "." )[ 0 ],
        max_recycles,
        model_id,
        n_struct_module_repeats,
        i,
        seed
      ) ) ) + ".pdb"

    to_pdb(
        job,
        outname,
        pred,
        result[ 'plddt' ],
        features_in[ 'residue_index' ]
    )

    # Important for multi-iteration runs
    template_pdb = outname

  return outname, result

def to_pdb(
    job: str,
    outname,
    pred, # type unknown but check?
    plddts, # type unknown but check?
    res_idx
  ) -> NoReturn:

  r""" Writes unrelaxed PDB to file

  Args:
    job: Jobname
    name: Model name
    pred: Actual prediction to write to PDB file
    plddts: Predicted errors
    res_idx: Residues to print (default=all; ???)
  """

  with open( outname, 'w' ) as outfile:
    outfile.write( alphafold.common.protein.to_pdb( pred ) )

  with open( f"b_{ outname }", "w" ) as outfile:
    for line in open( outname, "r" ).readlines():
      if line[ 0:6 ] == "ATOM  ":
        seq_id = int( line[ 22:26 ].strip() ) - 1
        seq_id = np.where( res_idx == seq_id )[ 0 ][ 0 ]
        outfile.write( "{}A{}{:6.2f}{}".format(
            line[ :21 ],
            line[ 22:60 ],
            plddts[ seq_id ],
            line[ 66: ] )
        )
  os.rename( f"b_{ outname }", outname )

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
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.model import modules
from alphafold.model import folding
from alphafold.model import quat_affine
from alphafold.data.tools import hhsearch

import jax.numpy as jnp
import jax

QUAT_AFFINE = None

NEW_INIT_COORDS = None

def monkey_patch_file(
    pdbfile: str,
    n_res: int
):
    n_xyz = np.zeros( ( 3, n_res ) )
    ca_xyz = np.zeros( ( 3, n_res ) )
    c_xyz = np.zeros( ( 3, n_res ) )

    new_coords = jnp.zeros( (
      n_res, residue_constants.atom_type_num, 3 ) )

    parser = Bio.PDB.PDBParser()
    struct = parser.get_structure("TEMP", pdbfile )


    for i, res in enumerate( struct.get_residues() ):
      n_xyz[ :, i ] = res[ "N" ].get_vector().get_array() / 10.
      ca_xyz[ :, i ] = res[ "CA" ].get_vector().get_array() / 10.
      c_xyz[ :, i ] = res[ "C" ].get_vector().get_array() / 10.

      for atom in res.get_atoms():
        atom_idx = residue_constants.atom_types.index( atom.get_id() )
        atom_xyz = atom.get_vector().get_array()
        new_coords = new_coords.at[ i, atom_idx, : ].set( atom_xyz ) / 10.

    NEW_INIT_COORDS = new_coords

    rot, trans = quat_affine.make_transform_from_reference(
        n_xyz=n_xyz.transpose(),
        ca_xyz=ca_xyz.transpose(),
        c_xyz=c_xyz.transpose()
      )
    QUAT_AFFINE = quat_affine.QuatAffine(
        quaternion=quat_affine.rot_to_quat( rot, unstack_inputs=True ),
        translation=trans,
        rotation=rot,
        unstack_inputs=True
      )

    def new_affine_monkey_patch( sequence_mask ):
      return QUAT_AFFINE

    def new_coords_monkey_patch( new_residues ):
      return NEW_INIT_COORDS

    # The actual monkey patch
    folding.generate_new_affine = new_affine_monkey_patch
    modules.init_new_coords = new_coords_monkey_patch


def predict_structure(
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

  f""" Predicts the structure.

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

  print( "Prediction parameters:" )
  print( f"\tUsing templates: { use_templates }" )
  print( f"\tMaximum number of MSA clusters: { max_msa_clusters }" )
  print( f"\tMaximum number of extra MSA clusters: { max_extra_msa }" )
  print( f"\tMaximum number of recycling iterations: { max_recycles }" )
  print( f"\tMinimum RMSD tolerance for early termination: { tol }" )

  if use_templates and template_paths is not None:
    util.randomize_templates( template_paths, move_prefix, n_template )
    tfeatures = util.mk_template( seq, a3m_lines, template_paths )
    tfeatures_in = tfeatures.features
    print( f"{ len( tfeatures.errors ) } errors with templates" )
    print( f"{ len( tfeatures.warnings ) } warnings with templates" )
    if len( tfeatures.errors ) > 0:
      print( "Errors:" )
      for error in tfeatures.errors:
        print( f"\t{ error }" )
    if len( tfeatures.warnings ) > 0:
      print( "Warnings:" )
      for warning in tfeatures.warnings:
        print( f"\t{ warning }" )
  else:
    #del template_paths # It is None if use_templates is false
    tfeatures_in = util.mk_mock_template( seq )

  # Assemble the dictionary of input features
  features_in = util.setup_features( seq, a3m_lines, tfeatures_in )

  # Run the models

 # model_id = random.randint( 1, 2 )
  model_id = 1
  model_id2 = 2
  print( f"Running model ID { model_id }" )

  params = alphafold.model.data.get_model_haiku_params(
      model_name=f"model_{ model_id }_ptm",
      data_dir="." )

  cfg = alphafold.model.config.model_config( f"model_{ model_id2 }_ptm" )

  # Here we provide config settings
  cfg.data.eval.num_ensemble = 1
  cfg.data.eval.max_msa_clusters = min(
      len( features_in[ "msa" ] ), max_msa_clusters )
  cfg.data.common.max_extra_msa = max( 1, min(
      len( features_in[ "msa" ] ) - max_msa_clusters, max_extra_msa )
  )
  cfg.model.recycle_tol = tol
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

  model_runner = alphafold.model.model.RunModel( cfg, params )

  features = model_runner.process_features(
      features_in,
      random_seed=random_seed
  )

  # Generate the model
  result = model_runner.predict( features )
  pred = alphafold.common.protein.from_prediction( features, result )

  to_pdb(
      job,
      idx,
      pred,
      result[ 'plddt' ],
      features_in[ 'residue_index' ]
  )
  if use_templates and template_paths is not None:
      util.reset_pdb70( template_paths, move_prefix )

  return result

def predict_structure_from_template(
    job: str,
    seq: str,
    idx: int,
    a3m_lines,
    template_pdb: str,
    n_models=1,
    random_seed=None,
    max_msa_clusters=32,
    max_extra_msa=64,
    tol=0,
    max_recycles=3,
    model_id=1
  ):

  f""" Predicts the structure.

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

  if random_seed is None:
    random_seed = random.randrange( sys.maxsize )

  print( "Prediction parameters:" )
  print( f"\tTemplate: { template_pdb }" )
  print( f"\tMaximum number of MSA clusters: { max_msa_clusters }" )
  print( f"\tMaximum number of extra MSA clusters: { max_extra_msa }" )
  print( f"\tMaximum number of recycling iterations: { max_recycles }" )
  print( f"\tMinimum RMSD tolerance for early termination: { tol }" )

  pdb = protein.from_pdb_string( util.pdb2str( template_pdb ) )

  tfeatures_in = {
    "template_aatype" : jax.nn.one_hot( pdb.aatype, 22 )[ : ][ None ],
    "template_all_atom_masks" : pdb.atom_mask[ : ][ None ],
    "template_all_atom_positions" : pdb.atom_positions[ :][ None ],
    "template_domain_names" : np.asarray( [ "None" ] ) }

  # Assemble the dictionary of input features
  features_in = util.setup_features( seq, a3m_lines, tfeatures_in )

  #model_id = random.randint( 1, 2 )
  print( f"Running model ID { model_id }" )

  params = alphafold.model.data.get_model_haiku_params(
      model_name=f"model_{ model_id }_ptm",
      data_dir="." )

  cfg = alphafold.model.config.model_config( f"model_{ model_id }_ptm" )

  # Here we provide config settings
  cfg.data.eval.num_ensemble = 1
  cfg.data.eval.max_msa_clusters = min(
      len( features_in[ "msa" ] ), max_msa_clusters )
  cfg.data.common.max_extra_msa = max( 1, min(
      len( features_in[ "msa" ] ) - max_msa_clusters, max_extra_msa )
  )
  cfg.model.recycle_tol = tol
  cfg.data.common.num_recycle = max_recycles
  cfg.model.num_recycle = max_recycles
  cfg.data.common.use_templates = True
  cfg.model.embeddings_and_evoformer.template.embed_torsion_angles = \
    True
  cfg.model.embeddings_and_evoformer.template.enabled = True
  cfg.data.common.reduce_msa_clusters_by_max_templates = True
  cfg.data.eval.subsample_templates = True


  #   Note that the structure module is recurrent, unlike the evoformer
  #   cfg.model.heads.structure_module.num_layer = 64

  model_runner = alphafold.model.model.RunModel( cfg, params )

  features = model_runner.process_features(
      features_in,
      random_seed=random_seed
  )

  # Generate the model
  result = model_runner.predict( features )
  pred = alphafold.common.protein.from_prediction( features, result )

  to_pdb(
      job,
      idx,
      pred,
      result[ 'plddt' ],
      features_in[ 'residue_index' ]
  )

  return result

def to_pdb(
    job: str,
    model_i: int,
    pred, # type unknown but check?
    plddts, # type unknown but check?
    res_idx
  ) -> NoReturn:

  f""" Writes unrelaxed PDB to file

  Args:
    job: Jobname
    model_i: Index number
    pred: Actual prediction to write to PDB file
    plddts: Predicted errors
    res_idx: Residues to print (default=all; ???)
  """

  outname = "_".join( map( str, ( job, "model", model_i ) ) ) + ".pdb"
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

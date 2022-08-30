import Bio.PDB
import numpy as np
import jax.numpy as jnp

from alphafold.common import residue_constants
from alphafold.model import quat_affine
from alphafold.model import modules
from alphafold.model import folding

# Objects for monkey patching
QUAT_AFFINE = None

NEW_INIT_COORDS = None

def monkey_patch_file(
    pdbfile: str,
    n_res: int
  ) -> NoReturn:
    r""" Function for "monkey-patching" functions in alphafold

    Note that this only works with the forked version available from
      delalamo/alphafold

    Args:
      pdbfile: Input PDB file
      n_res: Number of residues

    Output:
      None
    """
    n_xyz = np.zeros( ( 3, n_res ) )
    ca_xyz = np.zeros( ( 3, n_res ) )
    c_xyz = np.zeros( ( 3, n_res ) )

    new_coords = jnp.zeros( ( n_res, residue_constants.atom_type_num, 3 ) )

    struct = Bio.PDB.PDBParser().get_structure("TEMP", pdbfile )

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
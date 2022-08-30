import hashlib
import re
import os
import alphafold
import random
import numpy as np

from typing import NoReturn
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.data.tools import hhsearch

def pdb2str( pdbfile: str ) -> str:

  r""" Converts PDB file to string

  Credit to Sergey Ovchinnikov for writing this

  Args:
    pdbfile: String with PDB file to convert

  Output:
    String

  """
  lines = []
  for line in open( pdbfile, "r" ):
    if line[ :4 ] == "ATOM":
      lines.append( line )
  return "".join( lines )

def add_hash(
    jobname: str,
    seq: str
  ) -> str:
  
  r""" Generates a hash key for the job

  Args:
    jobname: Name of job
    seq: Amino acid sequence

  Output:
    hash key (five digits)

  """

  return jobname + "_" + hashlib.sha1( seq.encode() ).hexdigest()[ :5 ]

###############################

def process_seq(
    seq: str
  ) -> str:
  
  r""" Removes whitespace and non-alphabetic characters and
  returns an all-caps version of the string

  Args:
    seq: String with amino acid sequence

  Output:
    seq: Same string except uppercase and without non-alphabetic chars

  """

  return re.sub( r'[^a-zA-Z]', '', "".join( seq.split() ) ).upper()

###############################

def process_jobname(
    jobname: str
  ) -> str:
  
  r""" Gets rid of any non-alphanumeric characters in jobname

  Args:
    jobname: String with amino acid sequence

  Output:
    jobname: Same string without whitespace or non-alphanumeric chars

  """
  return re.sub( r"\W+", "", "".join( jobname.split() ) )

###############################

def write_fasta(
    filename: str,
    seq: str
  ) -> NoReturn:
  
  r"""Write fasta file with sequence; also writes single sequence MSA

  Args:
    filename: Target filename
  seq: Amino acid sequence

  Output:
    None

  """

  with open( filename, "w" ) as text_file:
    text_file.write( f">1\n{ seq }" )

###############################

def mk_mock_template(
    seq: str
  ) -> dict:
  
  r"""Generates mock templates that will not influence prediction
  
  Args:
    seq: Query sequence

  Output:
    Dictionary with blank/empty features

  """

  # Define constants
  lentype = alphafold.data.templates.residue_constants.atom_type_num
  lenseq = len( seq )

  # Since alphafold's model requires a template input
  # We create a blank example w/ zero input, confidence -1
  aatypes = np.array(
      alphafold.data.templates.residue_constants.sequence_to_onehot(
        "-" * lenseq,
        alphafold.data.templates.residue_constants.HHBLITS_AA_TO_ID
  ) )
  return {
      'template_all_atom_positions': np.zeros( ( lenseq, lentype, 3 ) )[ None ],
      'template_all_atom_masks': np.zeros( ( lenseq, lentype ) )[ None ],
      'template_sequence': [ f'none'.encode() ],
      'template_aatype': aatypes[ None ],
      'template_confidence_scores': np.full( lenseq, -1 )[ None ],
      'template_domain_names': [ f'none'.encode() ],
      'template_release_date': [ f'none'.encode() ]
  }

###############################

def mk_template(
    seq: str,
    a3m_lines = str,
    path = str
  ) -> dict:
  
  r""" Parses templates into features

  Args:
    a3m_lines: Lines form MMSeqs2 alignment
    path: Path to templates fetched using MMSeqs2

  Output:
    Dictionary with features

  """

  result = hhsearch.HHSearch(
      binary_path="hhsearch",
      databases=[ f"{ path }/pdb70" ]
  ).query( a3m_lines )

  return templates.HhsearchHitFeaturizer(
      mmcif_dir=path,
      max_template_date="2100-01-01",
      max_hits=20,
      kalign_binary_path="kalign",
      release_dates_path=None,
      obsolete_pdbs_path=None
  ).get_templates(
      query_sequence=seq,
      #query_pdb_code=None,
      #query_release_date=None,
      hits=alphafold.data.pipeline.parsers.parse_hhr( result )
  )

###############################

def setup_features(
    seq: str,
    a3m_lines: list,
    tfeatures_in: dict
  ) -> dict:
  
  r""" Set up features for alphafold

  Args:
    seq: Sequence (string)
    a3m_lines: Sequence alignment lines
    tfeatures_in: Template features

  Output:
    Alphafold features object

  """

  msa= pipeline.parsers.parse_a3m( a3m_lines )

  # Assemble the dictionary of input features
  return { **pipeline.make_sequence_features(
            sequence = seq,
            description = "none",
            num_res = len( seq )
          ),
          **pipeline.make_msa_features(
            msas = [ msa ]
          ),
          **tfeatures_in,
  }


###############################

def randomize_templates(
    template_path: str,
    prefix: str,
    n_template: int
  ) -> NoReturn:

  r""" Randomize templates for TM-Align (no longer used)

  Args:
    template_path: Path to templates
    prefix: Prefix to name file (for later retrieval)
    n_templates: Number of templates to fetch

  Output:
    None
    
  """
  
  pdb70 = template_path.split( "/" )[ 0 ] + "/pdb70.m8"
  lines = []
  with open( pdb70 ) as infile:
    for i, line in enumerate( infile ):
      if len( line.split() ) > 1 and i < 20: # max 20 templates
        lines.append( line )
  os.system( f"mv { pdb70 } { prefix }_pdb70.m8" )

  to_keep = []
  with open( pdb70, "w" ) as outfile:
    for line in random.sample( lines, min( len( lines ), n_template ) ):
      print( line )
      outfile.write( line )
      to_keep.append( line.split()[ 1 ] )

  for file in os.listdir( template_path ):
    if file.endswith( "ffindex" ):
      fullpath = os.path.join( template_path, file )
      newname = f"{ prefix }_{ file }"
      os.system( f"mv { fullpath } { newname }" )
      with open( newname ) as infile:
        with open( fullpath, "w" ) as outfile:
          for line in infile:
            if len( line.split() ) < 2:
              continue
            if line.split()[ 0 ] in to_keep:
              outfile.write( line )

def reset_pdb70(
    template_path: str,
    prefix: str
  ) -> NoReturn:

  r""" Return fetch templates (no longer needed)

  Args:
    template_path: Path to templates
    prefix: Prefix to name file (for later retrieval)

  Output:
    None
    
  """
  
  pdb70 = template_path.split( "/" )[ 0 ] + "/pdb70.m8"
  os.system( f"mv { prefix }_pdb70.m8 { pdb70 }" )

  for file in os.listdir():
    if file.endswith( "ffindex" ):
      newname = os.path.join( template_path, file[ len( prefix ) + 1: ] )
      os.system( f"mv { file } { newname }" )







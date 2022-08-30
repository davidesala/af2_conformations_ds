# If templates set to true (set to True to test):
#   Go through templates that were retrieved (cif)
#   Save a copy
#   Convert to PDB
#   Align to templates in LeuT_fold using TM-Align
#   Thread sequence through template (custom fxn)
#   Export to cif and replace the templates that were there

import Bio.PDB
import os
import random

# These are the only backbone atoms we are interested in copying
bb_atoms = [ "C", "CA", "N", "CB", "O" ]

# Map of single-digit to triple-digit residue types
restypes = { 	"A": "ALA",
				"C": "CYS",
				"D": "ASP",
				"E": "GLU",
				"F": "PHE",
				"G": "GLY",
				"H": "HIS",
				"I": "ILE",
				"K": "LYS",
				"L": "LEU",
				"M": "MET",
				"N": "ASN",
				"P": "PRO",
				"Q": "GLN",
				"R": "ARG",
				"S": "SER",
				"T": "THR",
				"V": "VAL",
				"W": "TRP",
				"Y": "TYR",
				"-": "---" }

def cif_to_pdb(
    path: str,
    cif_files: list,
    debug: bool = False
  ):
  
  r"""Converts a list of MMCIF files to PDB

  Args:
    path: Template path
    cif_files: List of files to convert
    debug: Whether to print debug statements

  Returns:
    None
  """

  parser = Bio.PDB.MMCIFParser()
  io = Bio.PDB.PDBIO()
  for file in cif_files:
    cif_file = os.path.join( path, file )
    cif = parser.get_structure( "TEMP", cif_file )
    filename = file.split( "." )[ 0 ]
    pdb_id = os.path.join( path, filename + ".pdb" )
    io.set_structure( cif )
    io.save( pdb_id )
    if debug:
      print( f"cif2pdb: { cif_file } { pdb_id }" )

def pdb_to_cif(
    path: str,
    pdb_files: list,
    debug: bool = False
  ):

  
  r"""Converts a list of PDB files to MMCIF
  Note: Chain is always set to A! This is corrected below

  Args:
    path: Template path
    pdb_files: List of files to convert
    debug: Whether to print debug statements

  Returns:
    None
  """

  out_cifs = []
  
  parser = Bio.PDB.PDBParser()
  io = Bio.PDB.MMCIFIO()
  for file in pdb_files:
    pdb_file = os.path.join( path, file )
    pdb = parser.get_structure( "TEMP", pdb_file )
    filename = file.split( "." )[ 0 ]
    cif_file = os.path.join( path, filename + ".cif" )
    io.set_structure( pdb )
    io.save( cif_file )
    if debug:
      print( f"pdb2cif: { pdb_file } { cif_file }" )
    out_cifs.append( cif_file )
  return out_cifs

def rand_rethread(
    pdb_file: str,
    tmalign: str,
    conf: str,
    leut_fold_path: str,
    chain: str,
    debug: bool = False
  ):
  
  r"""Randomly rethreads a template through a homolog
  Note: Chain is always set to A! This is corrected below

  Args:
    template_path: Template path
    pdb_file: PDB file with sequence to thread
    tmalign: Directory and name of TMAlign executable
    conf: Conformation (either "of" or "if")
    leut_fold_path: Directory with templates

  Returns:
    Name of output file
  """
  
  target = rand_template( os.path.join( leut_fold_path, "templates", conf ) )
  cmd = f"./{ tmalign } { pdb_file } { target }"
  x = os.popen( cmd ).read().splitlines()
  threadseq, targseq = x[ 18 ].rstrip(), x[ 20 ].rstrip() 
  tmscore1, tmscore2 = map( float, ( x[ 13 ][ 11:16 ], x[ 14 ][ 11:16 ] ) )
  
  if debug:
    print( f"\tThreading onto { target }" )

  # TM-Score check: Not used if less than 0.5 (different folds)
  if tmscore1 < 0.5 and tmscore2 < 0.5:
    return None

  thread(
    threadseq,
    targseq,
    target,
    pdb_file,
    chain
  )

def rand_template(
    target_path: str
  ):
  
  r"""Picks a template for threading at random

  Args:
    target_path: where to find the templates
  """

  files = []
  for file in os.listdir( target_path ):
    if file.endswith( ".pdb" ):
      files.append( file )
  return os.path.join( target_path, random.choice( files ) )

def thread(
    threadseq: str,
    targseq: str,
    targ_pdb: str,
    output_pdb: str,
    chain: str
  ):
  r"""Threads one sequence onto another

  Args:
    threadseq: Sequence to thread
    targseq: Target sequence
    targ_pdb: Target PDB file
    output_pdb: Where to dump new PDB

  Returns:
    None
  """

  assert len( threadseq ) == len( targseq )

  assert len( chain ) == 1

  resmap = []
  for threadres, targres in zip( threadseq, targseq ):
    if targres == "-":
      continue
    else:
      resmap.append( threadres )

  # Do the threading
  current_res = "X"
  current_resn = 0
  current_idx = 0
  with open( output_pdb, 'w' ) as outfile:
    with open( targ_pdb ) as infile:
      for line in infile:
        if not line.startswith( "ATOM" ):
          continue
        res = int( line[ 23:27 ] )
        atom = line[ 13:16 ].strip()
        if res > current_resn:
          current_resn = res
          current_idx += 1
        if current_idx >= len( resmap ):
          continue
        if targres not in restypes.keys():
          current_res = "---"
        else:
          current_res = restypes[ resmap[ current_idx ] ]
        if current_res == "---" or atom not in bb_atoms:
          continue
        if current_res == "GLY" and atom == "CB":
          continue
        outfile.write( line[ :17 ] )
        outfile.write( current_res )
        outfile.write( f" { chain }" )
        outfile.write( str( current_idx ).rjust( 4 ) )
        outfile.write( line[ 27: ] )

def combine_mmcifs(
  old_file: str,
  new_file: str
):
  
  atom_lines = []
  new_loop_lines = []
  with open( new_file ) as infile:
    for line in infile:
      if len( atom_lines ) == 0:
        if line.startswith( "loop_" ):
          new_loop_lines = [ line ]
        elif line.startswith( "_atom_site." ):
          new_loop_lines.append( line )
      if line.startswith( "ATOM" ):
        atom_lines.append( line )
  atom_lines = new_loop_lines + atom_lines
  
  del new_loop_lines
  
  os.system( f"rm { new_file }" )

  to_write = []
  reached_atom = False
  with open( new_file, "w" ) as outfile:
    with open( old_file ) as infile:
      for line in infile:
        if not reached_atom:
          if line.startswith( "ATOM" ):
            for item in reversed( to_write ):
              if item.startswith( "_atom_site." ):
                to_write.pop()
              elif item.startswith( "loop_" ):
                to_write.pop()
                break
            for item in to_write:
              outfile.write( item )
            for item in atom_lines:
              outfile.write( item )
            reached_atom = True
            # execute loop here
          else:
            to_write.append( line )
        elif not line.startswith( "ATOM" ) and not line.startswith( "HETATM" ):
          outfile.write( line )
  os.system( f"mv { new_file } { old_file }" )

def modify_templates(
    template_path: str,
    tmalign_exe: str,
    conf: str,
    leut_fold_path: str,
    debug: bool = False,
    max_templates: int = 20
  ):

  r"""Top-level function to mix up the templates

  Args:
    template_path: Template path
    tmalign_exe: Directory and name of TMAlign executable
    conf: Conformation (either "of" or "if")
    leut_fold_path: Directory with templates
    from_scratch: Whether to reinitialize templates from scratch
    debug: Whether to print debug statements

  Returns:
    None
  """

  assert conf in [ "of", "if" ]

  temp_name = "temp.pdb"

  cifparser = Bio.PDB.MMCIFParser()
  pdbparser = Bio.PDB.PDBParser()

  pdb70 = os.path.join( template_path.split( "/" )[ 0 ], "pdb70.m8" )
  with open( pdb70 ) as pdb70file:
    for i, line in enumerate( pdb70file ):
      if i >= max_templates:
        break
      if len( line ) < 10:
        continue
      pdb = line[ 4:8 ]
      chain = line[ 9 ]

      print( f"Modifying { pdb } chain { chain }" )

      filename = os.path.join( template_path, pdb.lower() + ".cif" )
      structure = cifparser.get_structure( "TEMP", filename )

      header = cifparser._mmcif_dict
      
      # Only one model is considered
      found_model = False
      for model in structure.get_models():

        pdbio = Bio.PDB.PDBIO()

        # Look through chains
        for modelchain in model.get_chains():
          if modelchain.get_id() == chain:
            pdbio.set_structure( modelchain )
            model.detach_child( chain )
            found_model = True
            break
        
        if found_model == False:
          continue

        # Save the PDB
        pdbio.save( temp_name )
        
        # TM-align and thread
        rand_rethread(
            temp_name,
            tmalign_exe,
            conf,
            leut_fold_path,
            chain,
            debug
          )

        # Re-import and re-save
        new_structure = pdbparser.get_structure( "TEMP", temp_name )
        chain_idx = ord( chain.lower() ) - 97
        model.insert( chain_idx, new_structure[ 0 ][ chain ] )
        if debug:
          print( f"\tWriting as chain { chain } (IDX: { chain_idx })" )
        cifio = Bio.PDB.MMCIFIO()
        cifio.set_structure( structure )
        cifio.save( temp_name )
        combine_mmcifs( filename, temp_name )
        break
      
      if found_model == False:
        raise Exception( f"Chain { chain } not found in { filename }!" )
        

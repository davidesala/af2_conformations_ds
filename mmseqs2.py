import requests
import os
import numpy as np
import time
import tarfile
from typing import NoReturn

class MMSeqs2Runner:

  r"""Runner object

  Fetches sequence alignment and templates from MMSeqs2 server
  Based on the function run_mmseqs2 from ColabFold (sokrypton/ColabFold)
  Version 62d7558c91a9809712b022faf9d91d8b183c328c

  Private variables:
  self.job: Job ID (five-char string)
  self.seq: Sequence to search
  self.seqs: List of sequence
  self.host_url: URL address to ping for data
  self.path: Path to use
  self.tarfile: Compressed file archive to download
  self.size: ??? For some reason ColabFold
  self.mmseqs2_id: ID
  """

  def __init__(
      self,
      job: str,
      seq: str,
      host_url="https://a3m.mmseqs.com",
      template_url="https://a3m-templates.mmseqs.com/template/",
      path_suffix="env"
    ):

    r"""Initialize runner object

    Args:
      job: Job name
      seq: Amino acid sequence
      host_url: Website to ping for sequence data
      template_url: Website to ping for template info
      path_suffix: Suffix for path info
    """

    assert len( seq ) > 0

    self.job = job
    self.seq = seq
    self.seqs = [ seq ]
    self.host_url = host_url
    self.template_url = template_url

    self.path = "_".join( ( job, path_suffix ) )
    if not os.path.isdir( self.path ):
      os.system( f"mkdir { self.path }" )

    self.tarfile = f'{ self.path }/out.tar.gz'

    self.size = 101

    self.mmseqs2_id = None

  def _submit(
      self
    ) -> dict:

    r"""Submit job to MMSeqs2 server

    Args:
      None
    """

    qs = [ f">{n+self.size}\n{s}\n" for n, s in enumerate( self.seqs ) ]
        
    res = requests.post(
      f'{ self.host_url }/ticket/msa',
      data={ 'q': "".join( qs ), 'mode': "env" } )
    try:
      out = res.json()

    except ValueError:
      out = { "status": "UNKNOWN" }

    return out

  def _status(
      self,
      idx
    ) -> dict:

    r"""Check status of job 

    Args:
      idx: Index assigned by MMSeqs2 server
    """

    res = requests.get( f'{ self.host_url }/ticket/{ idx }' )
    try:
      out = res.json()
    except ValueError:
      out = { "status": "UNKNOWN" }
    return out

  def _download(
      self,
      idx,
      path: str
    ):

    r"""Download job outputs

    Args:
      idx: Index assigned by MMSeqs2 server
      path: Path to download data to
    """

    # Set idx
    self.mmseqs2_id = idx

    res = requests.get( f'{ self.host_url }/result/download/{ idx }' )
    with open( path, "wb" ) as out:
      out.write( res.content )

  def _search_mmseqs2(
      self
    ) -> NoReturn:

    r"""Run the search and download results

    Args:
      None
    """

    # call mmseqs2 api
    redo = True
    
    # lets do it!
    if not os.path.isfile( self.tarfile ):
      while redo:
        out = self._submit()
        while out[ "status" ] in [ "UNKNOWN", "RATELIMIT" ]:
          # resubmit
          time.sleep( 5 + np.random.randint( 0, 5 ) )
          out = self._submit()

        while out[ "status" ] in [ "UNKNOWN", "RUNNING", "PENDING" ]:
          time.sleep( 5 + np.random.randint( 0, 5 ) )
          out = self._status(out[ "id" ])    
        
        if out[ "status" ] == "COMPLETE":
          redo = False
          
        if out[ "status" ] == "ERROR":
          redo = False
          msg = (
            "MMseqs2 API is giving errors. "
            "Please confirm your input is a valid protein sequence. "
            "If error persists, please try again an hour later. "
          )
          raise Exception( msg )
          
      # Download results
      self._download( out[ "id" ], self.tarfile )

  def process_templates(
      self,
      use_templates: bool,
      max_templates=20
    ):
    
    r"""Process templates and fetch from MMSeqs2 server

    Args:
      use_templates: True/False whether to use templates
      max_templates: Maximum number of templates to use
    """
    
    if not use_templates:
      return None

    #templates = {}
    print( "\t".join( ( "seq", "pdb", "cid", "evalue" ) ) )
    templates = []
    with open( f"{ self.path }/pdb70.m8", "r" ) as infile:
      for line in infile:
        sl = line.rstrip().split()
        t_idx = int( sl[ 0 ] )
        pdb = sl[ 1 ]
        templates.append( pdb )
        print( "\t".join( map( str, ( t_idx, pdb, sl[ 2 ], sl[ 10 ] ) ) ) )
      
    if len( templates ) == 0:
      raise Exception( "No templates found." )
    
    else:
      path = f"{ self.job }_env/templates_{ self.size }"
      if not os.path.isdir( path ):
        os.mkdir( path )
      pdbs = ",".join( templates[ :max_templates ] )
      fetch_cmd = (
        f"curl -v { self.template_url }{ pdbs }"
        f" | tar xzf - -C { path }/"
      ) 
      os.system( fetch_cmd )
      cp_cmd = (
        f"cp { path }/pdb70_a3m.ffindex "
        f"{ path }/pdb70_cs219.ffindex"
      )
      os.system( cp_cmd )
      touch_cmd = f"touch { path }/pdb70_cs219.ffdata"
      os.system( touch_cmd )
      return path

  def _process_alignment(
      self,
      a3m_files: list,
      token="\x00"
    ) -> dict:
    
    r"""
    Process sequence alignment

    Args:
      a3m_files: List of files to parse
      token: Token to look for when parsing
    """

    a3m_lines = {}
    for a3m_file in a3m_files:
      update = True
      m = None
      for line in open( a3m_file, "r" ):
        if len( line ) > 0:
          if token in line:
            line = line.replace( token, "" )
            update = True
          if line.startswith( ">" ) and update:
            m = int( line[ 1: ].rstrip() )
            update = False
            if m not in a3m_lines:
              a3m_lines[ m ] = []
          a3m_lines[ m ].append( line )
    
    return a3m_lines 

  def _seqs_from_alignment(
      self,
      alignment_info: dict
    ) -> str:
    
    r"""
    Retrieves sequence info from dictionary and returns as string

    Args:
      alignment_info: Information with alignment to be reformatted
    """

    for k, vals in alignment_info.items():
      return "".join( vals )

  def run_job(
      self,
      use_templates=True
    ):
    
    r"""
    Run sequence alignments using MMseqs2

    Args:
      use_templates: Whether to use templates
    """

    self._search_mmseqs2()

    # prep list of a3m files
    a3m_files = [ f"{ self.path }/uniref.a3m" ]
    a3m_files.append( f"{ self.path }/bfd.mgnify30.metaeuk30.smag30.a3m" )
    
    # extract a3m files
    if not os.path.isfile( a3m_files[ 0 ] ):
      with tarfile.open( self.tarfile ) as tar_gz:
        tar_gz.extractall( self.path )  

    alignment_info = self._process_alignment( a3m_files )
    templates_path = self.process_templates( use_templates )

    return self._seqs_from_alignment( alignment_info ), templates_path
#from google.colab import files
from . import util
import fileinput
import os

def import_custom_msa(
    a3m_file: str
  ) -> str:
  
  r"""
  Imports custom MSA and returns new sequence and MSA file

  Args:
    a3m_file: Filename of multiple sequence alignment

  Output:
    Amino acid sequence
  """

  custom_msa_dict = files.upload() # colab uploader
  custom_msa = list( custom_msa_dict.keys() )[ 0 ]
  header = 0
  for line in fileinput.FileInput( custom_msa, inplace=1 ):
    if line.startswith(">"):
        header += 1
    if line.startswith("#") or line.rstrip() == False:
      continue
    if line.startswith(">") == False and header == 1:
        seq = util.process_seq( line.rstrip() )
    print( line, end='' )

  os.rename( custom_msa, a3m_file )
  print( f"Moving { custom_msa } to { a3m_file }" )

  return seq

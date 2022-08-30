# -*- coding: utf-8 -*-
# Sequence and jobname
seq_in = 'GHMDDWEIPDGQITVGQRIGSGSFGTVYKGKWHGDVAVKMLNVTAPTPQQLQAFKNEVGVLRKTRHVNILQFMGYSTKPQLAIVTQWCEGSSLYHHLHASETKFEMKKLIDIARQTARGMDYLHAKSIIHRDLKSNNIFLHEDNTVKIGDFGLATVKSRWSGSHQFEQLSGSILWMAPEVIRMQDSNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQIIEMVGRGSLSPDLSKVRSNCPKRMKRLMAECLKKKRDERPSFPRILAEIEELARELSG'
job_in = 'T1195_1632_rec1' #@param {type:"string"}

# Type of multiple sequence alignment
#msa_mode = "MMseqs2" #@param ["MMseqs2 (UniRef+Environmental)", "MMseqs2 (UniRef only)","single_sequence" ]

max_msa = "16:32" #@param ["512:1024", "256:512", "128:256", "64:128", "32:64", "16:32"]
max_msa_clusters, max_extra_msa = [ int(x) for x in max_msa.split(":") ]

n_models = 50 #@param [1,3,5,15,50] {type:"raw"}
max_recycles = 1 #@param [1,3,5,15,50] {type:"raw"}
n_repeats = 8 #@param [8,16,32,64,128] {type:"raw"}
model_id = 1
model_params = 1

#from google.colab import files
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from scripts import mmseqs2
from scripts import predict
from scripts import thread
from scripts import upload_msa
from scripts import util
import multiprocessing
from absl import logging
logging.set_verbosity(logging.DEBUG)
#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#	print("Name:", gpu.name, "  Type:", gpu.device_type)



seq = util.process_seq( seq_in )
job = util.add_hash( util.process_jobname( job_in ), seq )

mmseqs2_runner = mmseqs2.MMSeqs2Runner( job, seq )
a3m_lines, template_paths = mmseqs2_runner.run_job(
    use_templates=False
  )



def run_prediction(job, seq, i, a3m_lines, model_id, model_params, max_msa_clusters, max_extra_msa, max_recycles, n_repeats):
  predict.predict_structure_no_templates(
      job,
      seq,
      i,
      a3m_lines,
      model_id=model_id,
      model_params=model_params,
      max_msa_clusters=max_msa_clusters,
      max_extra_msa=max_extra_msa,
      max_recycles=max_recycles,
      n_struct_module_repeats=n_repeats
  )
  print(f"Model {i} generated with modelID {model_id} and modelPARAMS {model_params}")



for i in range( n_models ):
  if i == 10:
    model_id = 1
    model_params = 2
  if i == 20:
    model_id = 2
    model_params = 1
  if i == 30:
    model_id = 2
    model_params = 2
  if i == 40:
    model_id = 2
    model_params = 3
  p = multiprocessing.Process(target=run_prediction(job, seq, i, a3m_lines, model_id, model_params, max_msa_clusters, max_extra_msa, max_recycles, n_repeats))
  p.start()
  p.join()


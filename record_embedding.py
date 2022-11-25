import os
import glob

if __name__ == '__main__':
  ref_wave = 'S04_M90_001'
  ## Get one embedding
  # os.system('python3 get_embedding.py --audio {}'.format(ref_wave))

  # Get all embedding
  excel_path = './corpus/metadata.csv'
  os.system("python get_embedding.py --audio {} --excel_path {}".format(ref_wave , excel_path))
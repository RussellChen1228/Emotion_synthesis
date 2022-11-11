import argparse
import os
import re
import numpy as np
from hparams import hparams
from synthesizer import Synthesizer
from util import audio

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  is_teacher_force = False
  reference_mel = None
  if args.mel_targets is not None:
    is_teacher_force = True
  reference_audio = './corpus/wavs/{}.wav'.format(args.audio)
  synth = Synthesizer(teacher_forcing_generating=is_teacher_force)
  synth.load(args.checkpoint, reference_audio)

  if args.excel_path is None:
    ref_wav = audio.load_wav(reference_audio)
    reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
    synth.record_embedding(args.text, reference_mel=reference_mel, audio_name=args.audio)
  
  else:
      with open(args.excel_path, encoding='utf-8') as sen:
        for line in sen:
          parts = line.strip().split('|')
          reference_audio = parts[0]
          text = parts[1]
          audio_path = './corpus/wavs/{}.wav'.format(reference_audio)
          ref_wav = audio.load_wav(audio_path)
          reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
          synth.record_embedding(text, reference_mel=reference_mel, audio_name=reference_audio)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default='./logs-tacotron/model.ckpt-0',  help='Path to model checkpoint')
  parser.add_argument('--text', default='\"thiau2 u3 hau3 nan5 ia1 uo3 hai5 srir2 fang2 tschi2 pa0\"')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--mel_targets', default=None, help='Mel-targets path, used when use teacher_force generation')
  parser.add_argument('--audio', required=True, default=None)
  parser.add_argument('--excel_path', default=None, help='Multi_synthesis')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()

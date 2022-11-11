import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import audio

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'wavs/'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  is_teacher_force = False
  mel_targets = args.mel_targets
  reference_mel = None
  if args.mel_targets is not None:
    is_teacher_force = True
    mel_targets = np.load(args.mel_targets)
  synth = Synthesizer(teacher_forcing_generating=is_teacher_force)
  synth.load(args.checkpoint, args.reference_audio, args.reference_embeddings)
  base_path = get_output_base_path(args.checkpoint)

  if args.reference_audio is not None:
    ref_wav = audio.load_wav(args.reference_audio)
    reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
    path = '%s%s.wav' % (base_path, os.path.splitext(os.path.basename(args.reference_audio))[0])
    alignment_path = '%s%s-align.png' % (base_path, os.path.splitext(os.path.basename(args.reference_audio))[0])
  else:
    if args.reference_embeddings is not None:
      if hparams.use_gst:
        print("*******************************")
        print('Use customize weights-embeddings.')
        print("*******************************")
        path = '%s%s.wav' % (base_path, os.path.splitext(os.path.basename(args.reference_embeddings))[0])
        alignment_path = '%s%s-align.png' % (base_path, os.path.splitext(os.path.basename(args.reference_embeddings))[0])
      else:
        raise ValueError("You must set the reference audio if you don't want to use GSTs.")
    else:
        print("*******************************")
        print("TODO: add style weights when there is no reference audio. Now we use random weights, " + 
               "which may generate unintelligible audio sometimes.")
        print("*******************************")
        path = '%srandomWeight.wav' % (base_path)
        alignment_path = '%s%s-align.png' % (base_path, 'randomWeight')


  if args.excel_path == '' or args.excel_path is None:
    with open(path, 'wb') as f:
      print('Synthesizing: %s' % args.text)
      print('Output wav file: %s' % path)
      print('Output alignments: %s' % alignment_path)
      f.write(synth.synthesize(args.text, mel_targets=mel_targets, reference_mel=reference_mel, alignment_path=alignment_path))
  
  else:
    with open(args.excel_path, encoding='utf-8') as sen:
      for line in sen:
        parts = line.strip().split('|')
        text = parts[1]
        path = './logs-tacotron/wavs/' + parts[0] + '.wav'
        with open(path, 'wb') as f:
          print('Output wav file: %s' % path)
          f.write(synth.synthesize(text, mel_targets=mel_targets, reference_mel=reference_mel, alignment_path=alignment_path))



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--text', required=True, default=None, help='Single test text sentence')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--reference_audio', default=None, help='Reference audio path')
  parser.add_argument('--reference_embeddings', default=None)
  parser.add_argument('--mel_targets', default=None, help='Mel-targets path, used when use teacher_force generation')
  parser.add_argument('--excel_path', default=None, help='Multi_synthesis')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()

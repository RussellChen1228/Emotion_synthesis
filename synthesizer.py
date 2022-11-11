import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio, plot
import textwrap

#整理embeddings格式
def adjust_embeddings(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        result = ''
        tmp = f.readlines()
        for i in range(len(tmp)):
            t1 = tmp[i].strip('[]').strip(' ')
            t2 = t1.replace('[', '').replace(']', '').replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ', '\n')
            result += t2
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result)

class Synthesizer:
  def __init__(self, teacher_forcing_generating=False):
    self.teacher_forcing_generating = teacher_forcing_generating
  def load(self, checkpoint_path, reference_mel=None, reference_embeddings=None, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths') 
    if reference_mel is not None:
      reference_mel = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'reference_mel')
    # Only used in teacher-forcing generating mode
    if self.teacher_forcing_generating:
      mel_targets = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_targets')
    else:
      mel_targets = None

    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths, mel_targets=mel_targets, reference_mel=reference_mel, reference_embeddings=reference_embeddings)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
      self.alignments = self.model.alignments[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())   

    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)
    

  def synthesize(self, text, mel_targets=None, reference_mel=None, alignment_path=None):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
    }
    if mel_targets is not None:
      mel_targets = np.expand_dims(mel_targets, 0)
      feed_dict.update({self.model.mel_targets: np.asarray(mel_targets, dtype=np.float32)})
    if reference_mel is not None:
      reference_mel = np.expand_dims(reference_mel, 0)
      feed_dict.update({self.model.reference_mel: np.asarray(reference_mel, dtype=np.float32)})

    wav, alignments = self.session.run([self.wav_output, self.alignments], feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    end_point = audio.find_endpoint(wav)
    wav = wav[:end_point]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    # n_frame = int(end_point / (hparams.frame_shift_ms / 1000* hparams.sample_rate)) + 1
    # text = '\n'.join(textwrap.wrap(text, 70, break_long_words=False))
    # plot.plot_alignment(alignments[:,:n_frame], alignment_path, info='%s' % (text))
    return out.getvalue()


  def record_embedding(self, text, reference_mel=None, audio_name=None, model_name='tacotron', mel_targets=None):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
    }
    if mel_targets is not None:
      mel_targets = np.expand_dims(mel_targets, 0)
      feed_dict.update({self.model.mel_targets: np.asarray(mel_targets, dtype=np.float32)})
    if reference_mel is not None:
      reference_mel = np.expand_dims(reference_mel, 0)
      feed_dict.update({self.model.reference_mel: np.asarray(reference_mel, dtype=np.float32)})
    filename_w = './embeddings/weight/' + audio_name +'.txt'
    filename_s = './embeddings/style/' + audio_name +'.txt'

    with open(filename_w, 'w', encoding='utf-8') as f:
      weights_embedding = ' '.join(str(i) for i in self.session.run(tf.get_default_graph().get_tensor_by_name('model/inference/Multihead-attention/mlp_attention_weights:0'), feed_dict=feed_dict))
      f.write(weights_embedding)
    adjust_embeddings(filename_w)

    with open(filename_s, 'w', encoding='utf-8') as f:
      style_embedding = ' '.join(str(i) for i in self.session.run(tf.get_default_graph().get_tensor_by_name('model/inference/Multihead-attention/Reshape_2:0'), feed_dict=feed_dict))
      f.write(style_embedding)
    adjust_embeddings(filename_s)

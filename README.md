# Emotion Synthesis (Base on GST-Tacotron)

A tensorflow implementation of the emotion synthesis.

Emotions are include neutral, cry, scary, sad, angry and happy.

## Quick Start:

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better performance, install with GPU support if it's available. This code works with TensorFlow 1.4.

3. Install requirements:

   ```
   pip install -r requirements.txt
   ```

### Training

1. **prepare a corpus:**

   - Put `metadata.scv` in `./corpus`: 
      - There are example in `./corpus/metadata.scv`
      - In each rows: `wav_file_name + | + text`
      - wav_file_name not include file extension

   - Put wavs file in  `./corpus/wavs`
      - Emotion set: E01:happy, E2:sad, E03:cry, E04:scary, E05:angry, SXX:neutral
      - Wav file name: EmotionSet_SpeakerID_TextNumber
      - Example: E01_M03_001.wav (Emotion=happy, SpeakerId=M03, TextNumber=001)
      - Example: S13_M13_005.wav (Emotion=neutral, SpeakerId=M13, TextNumber=005)

2. **Preprocess the data**
    
   ```
   python3 preprocess.py --dataset blizzard2013
   ```

3. **Train a model**

   ```
   python3 train.py
   ```
   
   The above command line will use default hyperparameters, which will train a model with cmudict-based phoneme sequence and 4-head multi-head sytle attention for global style tokens. If you set the `use_gst=False` in the hparams, it will train a model like Google's another paper [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"` . Hyperparameters should generally be set to the same values at both training and eval time.

4. **Get embedding**

   ```
   python3 get_embedding.py
   ```

   Record style embedding of all wavs, and save in `./embeddings/style`

5. **Show embedding**
   
   ```
   python3 show_embedding.py
   ```

   Show style embedding of all wavs in 2D, and you can choose reference wav to synthesis.

6. **Synthesize from a checkpoint**

   ```
   python3 genwav.py
   ```

   - The `./logs-tacotron/model.ckpt-0` is always the newest checkpoint. 
   - Also you can use excel file to automatic synthesis that the text in excel, you can set in `./genwav.py`.
   - If you want to synthesis more than one sentence, use excel file is mush faster.
   - Use excel is faster, because it only load the model once.

## Reference
  -  syang1993's implementation of gst-tacotron: https://github.com/syang1993/gst-tacotron
  -  Keithito's implementation of tacotron: https://github.com/keithito/tacotron
  -  Yuxuan Wang, Daisy Stanton, Yu Zhang, RJ Skerry-Ryan, Eric Battenberg, Joel Shor, Ying Xiao, Fei Ren, Ye Jia, Rif A. Saurous. 2018. [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017)
  - RJ Skerry-Ryan, Eric Battenberg, Ying Xiao, Yuxuan Wang, Daisy Stanton, Joel Shor, Ron J. Weiss, Rob Clark, Rif A. Saurous. 2018. [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).

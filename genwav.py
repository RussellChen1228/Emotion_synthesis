# _*_coding:utf-8_*_
import os
import time

if __name__=='__main__':
    taco_model = './logs-tacotron/model.ckpt-0'
    text = '\"jru5 kuo3 ni3 sciang3 iau2 uorn5 srorn1 ni3 tschy2 uorn5 hau3 liau3 \"'
    ref_wav = 'S04_F06_001'

    # Synthesie excel file
    # excel_path = './metadata.csv'
    # os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav --excel_path {}"
    # .format(taco_model, text, ref_wav,excel_path))

    start = time.time()
    os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav"
    .format(taco_model, text, ref_wav))
    end = time.time()
    print('cost time:', end - start)

# _*_coding:utf-8_*_
import os
import time

if __name__=='__main__':
    # The ./logs-tacotron/model.ckpt-0 is always newest.
    taco_model = './logs-tacotron/model.ckpt-0'
    # Text will be synthesised
    text = '\"i7 ka3 tscit8 kiann3 tai3 tsci3 khua2 ti3 scim7 kuann7 thau5\"'#伊共這件代誌掛佇心肝頭

    # Referance wav file for each emotion.
    nertual_ref = 'S04_F64_006'
    happy_ref   = 'E01_F64_010'
    sad_ref     = 'E02_F64_002'
    cry_ref     = 'E03_F64_002'
    scary_ref   = 'E04_F64_002'
    angry_ref   = 'E05_F64_002'

    start = time.time()

    ## Synthesis for only one sentence.
    os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav"
    .format(taco_model, text, nertual_ref))

    ## Synthesie excel file (./multi_syn.csv)
    # excel_path = './multi_syn.csv'
    # os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav --excel_path {}"
    # .format(taco_model, text, nertual_ref, excel_path))

    # Excel file will be synthesised for each emotion.
    nertual_excel_path  = './synthesis/netural.csv'
    happy_excel_path    = './synthesis/happy.csv'
    sad_excel_path      = './synthesis/sad.csv'
    cry_excel_path      = './synthesis/cry.csv'
    scary_excel_path    = './synthesis/scary.csv'
    angry_excel_path    = './synthesis/angry.csv'


    ## Synthesis nertual emotion and use excel file.
    # os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav --excel_path {}"
    # .format(taco_model, text, nertual_ref, nertual_excel_path))

    ## Synthesis happy emotion use and excel file.
    # os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav --excel_path {}"
    # .format(taco_model, text, happy_ref, happy_excel_path))

    ## Synthesis sad emotion and use excel file.
    # os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav --excel_path {}"
    # .format(taco_model, text, sad_ref, sad_excel_path))

    ## Synthesis cry emotion and use excel file.
    # os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav --excel_path {}"
    # .format(taco_model, text, cry_ref, cry_excel_path))

    ## Synthesis scary emotion and use excel file.
    # os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav --excel_path {}"
    # .format(taco_model, text, scary_ref, scary_excel_path))

    ## Synthesis angry emotion and use excel file.
    # os.system("python eval.py --checkpoint {} --text {} --reference_audio ./corpus/wavs/{}.wav --excel_path {}"
    # .format(taco_model, text, angry_ref, angry_excel_path))
    
    end = time.time()
    print('cost time:', end - start)

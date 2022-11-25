import glob

file_name = list()
with open('./corpus/metadata.csv', encoding='utf-8') as sen:
      for line in sen:
        parts = line.strip().split('|')
        file_name.append(parts[0])

for file in glob.glob('./corpus/wavs/*.wav'):
    file = file.replace('./corpus/wavs/', '').replace('.wav', '')
    if file not in file_name:
        print(file)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import glob

emotion_encoding_table = {
    'S' : 0,
    'E01' : 1,
    'E02' : 2,
    'E03':3, 
    'E04':4,
    'E04' : 5,
}

emotion_marker_table = {
    0 : ['neutral', 'black'] ,
    1 : ['happy', 'y'] ,
    2 : ['sad', 'b'], 
    3 : ['cry', 'o'],
    4 : ['scary', 'g'],
    5 : ['angry', 'r'] ,
}

def showEmbedding(speaker_id:str):
    names= list()
    emotions = list()
    embeddings=list()
    for data in glob.glob(r'./embeddings/style/*{}*.txt'.format(speaker_id)):
        embedding = np.loadtxt(data, delimiter=r' ')
        embedding = embedding.reshape([1, 256])
        embeddings.append(embedding)
        corpus_set = data.split('_')[0].split('/')[-1]
        # if S in the name of set , it's neutral set.
        if 'S' in corpus_set:
            corpus_set = 'S'

        emotions.append(emotion_encoding_table[corpus_set])
        name = data.split('_',1)[1].replace('.txt','')
        names.append(name)

    for idx, item in enumerate(embeddings):
        if idx == 0:
            pass
        elif idx == 1:
            c = np.concatenate((embeddings[0], embeddings[1]), axis = 0)            
        else:
            c = np.concatenate((c, item), axis = 0)

    X_tsne = manifold.TSNE(n_components=2, perplexity=350, early_exaggeration=16,
    init='pca', random_state=21, n_iter=1000).fit_transform(c)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize

    for i in range(X_norm.shape[0]):
        emotion_2d = emotions[i]
        plt.text(X_norm[i, 0], X_norm[i, 1], names[i], color=emotion_marker_table[emotion_2d][1], fontdict={'weight': 'bold', 'size': 9})

    plt.xticks()
    plt.yticks()
    plt.show()

if __name__ == '__main__':
    speaker_id = 'M12'
    showEmbedding(speaker_id)
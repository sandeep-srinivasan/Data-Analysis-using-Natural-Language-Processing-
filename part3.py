import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import os
import re
import string
import csv
from collections import OrderedDict
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def show_plot():
    import numpy as np
    import matplotlib
    matplotlib.get_backend()
    # matplotlib.use('TkAgg')
    # matplotlib.use('GTK3Cairo')
    # matplotlib.use('GTKAgg')
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt

    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    plt.show()


def evaluate_clutering(X, labels_true, labels_pred, algorithm = 'null', print_flag=True):
    a = metrics.adjusted_rand_score(labels_true, labels_pred)
    b = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    c = metrics.homogeneity_score(labels_true, labels_pred)
    d = metrics.completeness_score(labels_true, labels_pred)

    if algorithm!='null':
        loss = -algorithm.score(X)
    else:
        loss = 0
    if print_flag == True:
        print('adjusted rand score is %s' % a)
        print('adjusted mutual information is %s' % b)
        print('homogeneity_score is %s' % c)
        print('completeness_score is %s' % d)
    return np.array([a,b,c,d, loss])

def viz_cluster_qualify(data, title, name):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    red_patch = mpatches.Patch(color='red', label='adjusted rand score')
    blue_patch = mpatches.Patch(color='blue', label='adjusted mutual info')
    green_patch = mpatches.Patch(color='green', label='homogenity_score')

    # data = np.random.rand(10,5)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title(title)
    plt.legend(handles=[red_patch, blue_patch, green_patch])

    x = range(np.shape(data)[0])
    ax1.plot(x, data[:,0], 'r-', x, data[:, 1], 'b-', x, data[:,2], 'g-') #, x, data[:,3], 'g--')
    (bottom, top) = plt.xlim()
    plt.xlim((2, top))
    # ax2 = fig.add_subplot(212)
    # ax2.plot(x[2:], data[2:, 4],'r-')

    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/tmp/pycharm_project_54/image/'
    plt.savefig(direct + name + '.png')
    plt.close()

def kmeans_(n_cluster):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_jobs=-1).fit(X)
    return evaluate_clutering(X, Y1, kmeans.labels_, algorithm=kmeans)
   
def hierachical_clustering(n_cluster):
    assert 'X' in globals().keys()
    assert 'Y1' in globals().keys()
    hierachical_clustering = AgglomerativeClustering(n_clusters=n_cluster)
    return evaluate_clutering(X, Y1, hierachical_clustering.fit_predict(X), algorithm='null')


if __name__ == '__main__':
    # read all the articles
    #articles = read_articles(n=22); n = len(articles)
    #print('Finish reading %s the articles'%len(articles))

    # compute tf-idf
    #from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
    #corpus = [concatenate_lines(articles[i]) for i in range(len(articles))]
    #vectorizer = CountVectorizer()
   # tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    #tfidf_transformer.fit(vectorizer.fit_transform(corpus))
    #print('Finish computing tf-idf')

    #tf_idf_vector = tfidf_transformer.transform(vectorizer.transform(corpus))
    #top_n_words_feauture = top_n_words(tf_idf_vector, n=100)
    #print('Before removing', np.shape(top_n_words_feauture))
   # top_n_words_feauture = remove_zero_col(top_n_words_feauture)
    #print('After removing', np.shape(top_n_words_feauture))
    #from sklearn.decomposition import TruncatedSVD

    data_words = np.loadtxt(open("article_words.csv", "rb"), delimiter=",")
    #print (data)

    pca = PCA(n_components=120)
    pca.fit(data_words)
    print('Start PCA')
    data_w_reduced = pca.transform(data_words)
    print('Finish PCA')

    data_topics = np.loadtxt(open("article_topics.csv", "rb"), delimiter=",")


    test_k_vals = [5, 10, 25, 50, 100]
    for i in test_k_vals:
        kmeans = KMeans(n_clusters=i, random_state=0, n_jobs=-1).fit(data_w_reduced)
        print (kmeans)    
        #return evaluate_clutering(X, topics, kmeans.labels_, algorithm=kmeans)
        '''a = metrics.adjusted_rand_score(labels_true, labels_pred)
        b = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        c = metrics.homogeneity_score(labels_true, labels_pred)
        d = metrics.completeness_score(labels_true, labels_pred)

        if algorithm!='null':
        	loss = -algorithm.score(X)
        else:
        	loss = 0
        if print_flag == True:
        	print('adjusted rand score is %s' % a)
        	print('adjusted mutual information is %s' % b)
        	print('homogeneity_score is %s' % c)
        	print('completeness_score is %s' % d)
        return np.array([a,b,c,d, loss])'''


    ''' Y1 = getY_places(articles).toarray()
    Y1 = np.array([y.argmax() for y in Y1])
    Y2 = getY_topics(articles).toarray()
    Y2 = np.array([y.argmax() for y in Y2])
    print('Finish computing tf-idf and labels')

    # clustering
    X = tf_idf_vector_reduced
    n_qualify_measure = 5
    n_center_possibilities = 30 '''

    ''' kmeans_data = Parallel(n_jobs=-1)(delayed(kmeans_)(n_cluster) for n_cluster in range(2,n_center_possibilities))
    kmeans_data = np.vstack(kmeans_data)

    hierachical_clustering_data = Parallel(n_jobs=-1)(delayed(hierachical_clustering)(n_cluster) for n_cluster in range(2, n_center_possibilities))
    hierachical_clustering_data = np.vstack(hierachical_clustering_data)

    viz_cluster_qualify(kmeans_data, 'kmeans: tf vector for places', 'kmeans-places-lse')
    viz_cluster_qualify(hierachical_clustering_data, 'AgglomerativeClustering: tf vector for places', 'hiereachical-places-lse')

    '''
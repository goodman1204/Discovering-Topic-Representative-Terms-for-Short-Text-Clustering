import json
import os
from sklearn import metrics
from itertools import combinations
from collections import defaultdict,Counter
import log
import gensim
import numpy as np


def read_label(filepath,logger):
    logger.info("loadding data {}".format(filepath))
    fp = open(filepath)

    labels_true = []
    docs = []
    for line in fp:
        parsed = json.loads(line)
        docs.append(parsed['text'].strip().split())
        labels_true.append(int(parsed['cluster']))

    return labels_true


def read_document(filepath,logger):

    logger.info("open dataset")
    fp = open(filepath)
    doc = []
    for line in fp:
        parsed_line = json.loads(line)
        text = parsed_line['text'].strip().split()
        doc.append(text)

    return doc


def read_docs_labels(filepath,logger):

    logger.info("open dataset: {}".format(filepath))
    fp = open(filepath)
    doc = []
    labels_true = []
    for line in fp:
        parsed_line = json.loads(line)
        text = parsed_line['text'].strip().split()
        doc.append(text)
        labels_true.append(int(parsed_line['cluster']))

    return doc, labels_true


def computing_cooccurence(docs):
    """
    docs=[[word1,word2,word3],[word1,word2,word3],...]
    """


    word_cooccurrence = defaultdict(lambda: defaultdict(int))
    word_freq = defaultdict(int)

    for d in docs:
        word_comb = combinations(d, 2)
        for wordpair in word_comb:
            word1 = wordpair[0]
            word2 = wordpair[1]
            word_cooccurrence[word1][word2] += 1
            word_cooccurrence[word2][word1] += 1

        for word in d:
            word_freq[word] += 1

    # word_cooccurrence_sorted=defaultdict(list)

    # for word in word_cooccurrence.keys():

        # temp=[item for item in word_cooccurrence[word].items()]
        # temp.sort(key=lambda x:x[1],reverse=True)
        # word_cooccurrence_sorted[word]=temp

    # return word_cooccurrence,word_freq,word_cooccurrence_sorted
    return word_freq, word_cooccurrence


def load_word_statistics(filepath):

    with open(filepath) as fp:

        word_statistics = json.load(fp)
        return word_statistics


def load_keyword_group(filepath):

    with open(filepath) as fp:

        keyword_group = json.load(fp)
        return keyword_group


def json_dump(filepath, docs, labels):

    with open(filepath, 'w') as wp:
        for index, doc in enumerate(docs):
            d = {"text": " ".join(
                doc), "cluster": labels[index], "id": index + 1}
            wp.write("{}\n".format(json.dumps(d)))

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def clustering_evaluation(labels_true, labels,logger):
    logger.info("------------------------clustering result-----------------------------")
    logger.info("original dataset length:{},pred dataset length:{}".format(
        len(labels_true), len(labels)))
    logger.info('number of clusters in dataset: %d' % len(set(labels_true)))
    logger.info('number of clusters estimated: %d' % len(set(labels)))
    logger.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    logger.info("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    logger.info("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    logger.info("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    logger.info("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    logger.info("Normalized Mutual Information: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
    logger.info("Purity Score: %0.3f" % purity_score(labels_true, labels))
    logger.info("------------------------end ----------------------------------------")
    return (metrics.homogeneity_score(labels_true, labels), metrics.completeness_score(labels_true, labels), metrics.v_measure_score(
        labels_true, labels), metrics.adjusted_rand_score(labels_true, labels), metrics.adjusted_mutual_info_score(labels_true, labels),metrics.normalized_mutual_info_score(labels_true,labels),purity_score(labels_true, labels))


def analysis_predicted_label(
        labels_true, labels_predicted, docs, keyword_group, output_dir, dataset_name, args):

    with open(os.path.join(output_dir, "{}_base{}_batch{}_predicted_labels".format(dataset_name, args.base_size, args.batch_size)), "w") as wp:
        counter = 0
        for doc in docs:
            wp.write("{}\t{}\t{}\n".format(
                labels_true[counter], doc, labels_predicted[counter]))
            counter += 1

    with open(os.path.join(output_dir, "{}_base{}_batch{}_keyword_group".format(dataset_name, args.base_size, args.batch_size)), "w") as wp:

        for key in keyword_group.keys():
            wp.write("{}---{}\n".format(key, keyword_group[key]))


def purity_test(docs, labels_true, predicted_labels,logger,args):

    label_dict = defaultdict(list) #original labels in the newly predicted clusters
    label_doc_dict = defaultdict(list) #original docs and its labels in the newly predicted clusters

    for i in range(len(labels_true)):
        label_dict[predicted_labels[i]].append(labels_true[i])
        label_doc_dict[predicted_labels[i]].append((i+1,labels_true[i]))

    temp_list = []
    for key in label_dict.keys():
        cluster_size = len(label_dict[key]) #predicted clusters' size
        label_length = len(set(label_dict[key])) #orginal label numbers in a predicted cluster
        temp_list.append((key, cluster_size, label_length,Counter(label_dict[key]))) # predicted cluster, cluster size, original label size, original clusters' size in the predicted cluster

    temp_list.sort(key=lambda x: x[1], reverse=True)

    for item in temp_list:
        logger.info("predicted cluster:{0}, cluster size:{1}, original cluster numbers :{2}\n\t doc ratio for clusters:{3}\n\t\tOrignial doc's cluster:{4}".format(item[0], item[1], item[2],item[3],label_doc_dict[item[0]]))

    doc_length = len(docs)
    doc_label = labels_true

    basename = os.path.basename(args.dataset_dir)
    for item in temp_list:
        if item[1]>300:
            max_key=None
            max_value =0
            for key in item[3].keys():
                if item[3][key]>max_value:
                    max_key = key
                    max_value = item[3][key]

            for doc_id,label in label_doc_dict[item[0]]:
                if item[3][label] > 1:
                    doc_label[int(doc_id)-1]=max_key

    json_dump('{}_changed'.format(basename), docs, doc_label)



def tf_vector_computing(filepath,logger):
    logger.info("loadding data {}".format(filepath))
    fp = open(filepath)
    labels_true = []

    docs = []

    for line in fp:
        parsed = json.loads(line)
        docs.append(parsed['text'].strip().split())
        labels_true.append(int(parsed['cluster']))

    doc_size = len(docs)
    # x=np.zeros((doc_size,doc_size))

    logger.info("computing tf vetors")

    dictionary = gensim.corpora.Dictionary(docs)
    corpus_doc2bow = [dictionary.doc2bow(item) for item in docs]
    # tfidfmodel = gensim.models.tfidfmodel.TfidfModel(corpus_doc2bow)
    vector_dimension = len(dictionary)
    logger.info("vector_dimension:{}".format(vector_dimension))
    document_size = len(corpus_doc2bow)

    document_vector = []
    for doc in corpus_doc2bow:

        tfidf_vector = [0]*vector_dimension
        for item in doc:
            tfidf_vector[item[0]] = item[1]

        document_vector.append(tfidf_vector)

    data_vector = np.array(document_vector)

    logger.info("finishing tf vector computing")

    return data_vector


def vocab_size_computing(docs):

    vocab_size=set()

    for doc in docs:
        for word in doc:
            vocab_size.add(word)

    return len(vocab_size)












import networkx as nx
import argparse
import json
import sys
import os
import time
import log
import pickle
from itertools import combinations
import numpy as np

from collections import *
from utilities import read_document, computing_cooccurence, load_word_statistics,clustering_evaluation, purity_test, read_docs_labels

from sklearn.cluster import *

logger=None

CORE_NODE_COOC = 0.5
CORE_NODE_PORTION = 0.1
EDGE_COOC = 3


def construct_word_graph(word_cooccurrence, word_freq):

    logger.info("contruct word graph")
    g = nx.Graph()

    # for word in word_freq.keys():
    # g.add_node(word,freq=word_freq[word])
    filter_word_coocc = defaultdict(lambda: defaultdict(int))

    edge_list = []

    filter_word_list = []


    for word1 in word_cooccurrence.keys():

        if word1 in filter_word_list:
            continue

        for word2 in word_cooccurrence[word1].keys():

            if word2 in filter_word_list:
                continue

            if word_cooccurrence[word1][word2]>=EDGE_COOC:
                edge_list.append((word1,word2,{'weight':word_cooccurrence[word1][word2]}))

    g.add_edges_from(list(edge_list))

    logger.info("node length:{}, edge length:{}".format(
        len(g.nodes), len(g.edges)))
    return g


def keyword_group_discovery_sorted(graph, word_freq, word_cooccurrence,args):

    """
    sorted method for keyword group discovery

    """

    core_node_dict = defaultdict(list)  # store core node and its neighbours
    # store processed node, such as core node, connected core node.
    processed_nodes = defaultdict(int)

    # sort node according to their node weight
    node_list = [(node, word_freq[node]) for node in graph.nodes]
    node_list.sort(key=lambda x: x[1], reverse=True)



    for node, node_weight in node_list:



        if processed_nodes[node]:
            continue

        node_freq = word_freq[node]

        core_node_flag = True


        for nn in graph[node]:
            if word_freq[nn] > node_freq and not processed_nodes[nn]:
                core_node_flag = False
                break

        if not core_node_flag:
            continue

        # logger.debug("seed node:{}".format(node))

        processed_nodes[node] = 1
        core_node_dict[node].append(node)

        for nn in graph[node]:
#-------------------------------------------------------------------------------
            # judge if the neigbour of core node could be a keyword
            if word_freq[nn]/node_freq >= CORE_NODE_PORTION and word_cooccurrence[nn][node]/word_freq[nn] >= CORE_NODE_COOC:
                # logger.debug("\t\taffiliated node:{}".format(nn))
                core_node_dict[node].append(nn)
                processed_nodes[nn]=1 #mark the neigbour node of core node has been processed
#-------------------------------------------------------------------------------------

        # round_counter+=1
        if len(core_node_dict[node]) == 1:
            core_node_dict.pop(node)
        else:
            logger.info(core_node_dict[node])

    logger.info("keyword_group length:{}".format(len(core_node_dict)))


    return core_node_dict

def keyword_group_clustering(docs, word_freq, keyword_group,logger):
    """
    predict cluster label of short text based on keyword group mined from short text dataset.
    """


    doc_size = len(docs)

    sub_doc_size = int(doc_size/10)


    kw_index = 0
    kw_group = defaultdict(list)

    for key in keyword_group.keys():
        for word in keyword_group[key]:
            kw_group[kw_index].append(word)

        kw_index += 1


    logger.info("predict document labels")
    predict_labels = []
    # for saving all the document weight on each keyword group
    doc_weight = defaultdict(list)
    doc_weight_vector = []  # for saving all the document weight on each keyword group

    for index, doc in enumerate(docs):

        doc_set = set(doc)
        kw_group_weight = []  # for saving the weight of document on each keyword group
        # for saving the weight of document on each keyword group in a vector, then using it on kmeans
        kw_group_weight_vector = []

        max_weight = 0
        label_index = None

        for kw_index in kw_group.keys():
            kw_set = set(kw_group[kw_index])

            common_words = doc_set.intersection(kw_set)

            weight = len(common_words)/len(doc_set)  # jaccard similarity

            if weight > max_weight:
                max_weight = weight
                label_index = kw_index

        if label_index != None:
            # kw_group_weight.sort(key=lambda x: x[1], reverse=True)
            predict_labels.append(kw_group[label_index][0])
        else:
            # the cluster index of doc can not be predicted
            predict_labels.append(-1)

        #----end------------#
        if (index+1)%sub_doc_size == 0:
            logger.info("{:.3f} percent has been processed".format((index+1)/doc_size))

    return predict_labels


def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument('-d', '--dataset_dir', type=str, required=True, help='path to the corpus')
    parser.add_argument('-ws', '--word_statistics_dir', type=str, help='path to the corpus word statistics (optional)')
    parser.add_argument('-p', '--purity',  help='predicted label purity test (optional)',action = 'store_true')
    # parser.add_argument('--alpha', type=float, required=True, help='co-occurrence ratio to the frequency of affilaited node')
    parser.add_argument('--theta', type=float, required=True, help='co-occurrence ratio to the frequency of affilaited node')
    parser.add_argument('--delta', type=float, required=True, help='frequency ratio of affiliated node to seed node')
    # parser.add_argument('--beta', type=float, required=True, help='frequency ratio of affiliated node to seed node')
    parser.add_argument('--gamma', type=int, required=True, help='to filter edges in word graph')
    parser.add_argument('--num_processes', type=int,  help='number of processors for clustering',default=1)
    parser.add_argument('-r','--repeat', action='store_true', help='repeat of clustering')
    parser.add_argument('-t','--target_number', type=int,  help='keyword group numbers')
    args, unknown = parser.parse_known_args()

    return args


def main(args):

    start =time.time()
    docs, labels_true = read_docs_labels(args.dataset_dir,logger)

    if args.word_statistics_dir:
        logger.info("word_statistics")
        ws = load_word_statistics(args.word_statistics_dir)
        word_cooccurrence = ws['word_cooccurrence']
        word_freq = ws['word_freq']
        word_cooccurrence_sorted = ws['word_cooc_sorted']
    else:
        word_freq, word_cooccurrence= computing_cooccurence(docs)

    logger.info("parameters: gamma:{}, delta:{:.1f},theta:{:.1f}".format(args.gamma, args.delta, args.theta))

    global CORE_NODE_COOC
    global CORE_NODE_PORTION
    global EDGE_COOC

    CORE_NODE_COOC = args.theta # f(w,v)/f(v)
    CORE_NODE_PORTION = args.delta # f(v)/f(w)
    EDGE_COOC = args.gamma

    graph = construct_word_graph(word_cooccurrence, word_freq)
    stage1  = time.time()

    core_node_dict = keyword_group_discovery_sorted(graph, word_freq, word_cooccurrence,args)

    stage2 = time.time()

    predict_labels= keyword_group_clustering(docs, word_freq, core_node_dict,logger)

    stage3 = time.time()
    end = time.time()

    result = clustering_evaluation(labels_true, predict_labels,logger)

    basename = os.path.basename(args.dataset_dir)
    logger.info("saving predicted labels")
    with open("{}_predict_labels_{}_{:.1f}_{:.1f}.json".format(basename, EDGE_COOC, CORE_NODE_PORTION, CORE_NODE_COOC), 'w') as wp:
        for index, value in enumerate(predict_labels):
            wp.write("{}\n".format(value))

    logger.info("saving keyword group")
    with open("{}_keyword_group_{}_{:.1f}_{:.1f}.json".format(basename, EDGE_COOC, CORE_NODE_PORTION, CORE_NODE_COOC), 'w') as wp:
        json.dump(core_node_dict, wp)

    logger.info("saving word graph")
    with open("{}_word_graph_{}_{:.1f}_{:.1f}.pickle".format(basename, EDGE_COOC, CORE_NODE_PORTION, CORE_NODE_COOC), 'wb') as wp:
        pickle.dump(graph, wp)

    logger.info("start time: {}".format(start))
    logger.info("stage1 time cost:{:.3f}".format(stage1- start))
    logger.info("stage2 time cost:{:.3f}".format(stage2- stage1))
    logger.info("stage3 time cost:{:.3f}".format(stage3-stage2))
    logger.info("processing time cost:{:.3f}".format(end-start))
    logger.info("end time: {}".format(end))

    if args.purity:
        purity_test(docs,labels_true,predict_labels,logger,args)


    logger.info("all done")
    return result

if __name__ == "__main__":

    args = parse_args()
    basename=os.path.basename(args.dataset_dir)
    logger= log.get_logger(__file__,"{}_{}_{}_{}_{}_log".format(basename, args.gamma, args.theta, args.delta, args.num_processes))

    if not args.repeat:
        main(args)
    else:
        result_list = []
        for r in range(10):
            result = main(args)
            result_list.append(result)
        x = np.array(result_list)
        logger.info(np.mean(x, axis=0))
        logger.info(np.std(x, axis=0))


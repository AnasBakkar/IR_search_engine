import json
import math
from sklearn.cluster import KMeans
from fastapi.middleware.cors import CORSMiddleware
from autocorrect import Speller
import unicodedata
from fastapi import FastAPI
from nltk.corpus import stopwords, wordnet
import ir_datasets
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import np as np
from deep_translator import GoogleTranslator
import os.path

queriers = ir_datasets.load("antique/test")
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------- first req ---------------------------------

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
print('qrels ', queriers)
stop_chars = [';', ':', '!', "*", '.', ',', ')', "'s", '(', '?', '\',''/', '', '...', '..', '``']


def part_of_speech_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def process_dataset(dataset_index):
    raw_docs = {}
    corpus = {}
    # nlp = spacy.load("en_core_web_sm")
    if dataset_index == 0:
        dataset = ir_datasets.load("antique")
        processing_file = "processed_corpus.json"
        raw_file = "corpus.json"
    if os.path.isfile(processing_file):
        with open(processing_file, "r") as file:
            # if file:
            print(file)
            corpus = json.load(file)
        if os.path.isfile(raw_file):
            with open(raw_file, "r") as file2:
                raw_docs = json.load(file2)
    else:
        with open(dataset.docs_path(), encoding='utf-8') as f:
            i = 0
            #     print('stop words ', stop_words)
            for line in f:
                text = line[10:]
                raw_docs[line[0:9]] = text
                i += 1
                print('raw doc ', text)
                processed_text1 = text.lower().strip()
                print('lower text doc ', processed_text1)
                processed_text2 = word_tokenize(processed_text1)
                print('tokenized doc ', processed_text2)
                filtered_sentence = []
                processed_text3 = filter(lambda k: k not in stop_chars, processed_text2)
                # print('filtered from stop chars doc ', list(processed_text3))
                for w in processed_text3:
                    if w not in stop_words:
                        # word = ps.stem(w)
                        filtered_sentence.append(w)
                # print('fil ', filtered_sentence)
                pos_tagged = nltk.pos_tag(filtered_sentence)
                wordnet_tagged = list(map(lambda x: (x[0], part_of_speech_tagger(x[1])), pos_tagged))
                # print("wordnet_tagged :", wordnet_tagged)

                lemmatized_sentence = []
                for word, tag in wordnet_tagged:
                    if tag is None:
                        lemmatized_sentence.append(word)
                    else:
                        # print('word before lemmatize ', word)
                        lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
                        # print('after ', lemmatizer.lemmatize(word, tag), tag, word)
                # processed_text4 = nlp(lemmatized_sentence)
                corpus[line[0:9]] = lemmatized_sentence
                print('lemmatized & filterd doc ', corpus[line[0:9]])
                if i == 1000:
                    break

            raw_storage_file = open('corpus.json', 'w')
            json.dump(raw_docs, raw_storage_file)
            raw_storage_file.close()
            storage_file = open('processed_corpus.json', 'w')
            json.dump(corpus, storage_file)
            storage_file.close()
    return corpus, raw_docs


# -------------------------------- second & third req ----------------------------------

def create_inverted_index(corpus):
    inverted_index = defaultdict(list)
    for doc_id, doc_content in corpus.items():
        for term in doc_content:
            inverted_index[term].append(doc_id)
            j = 0
            for s in doc_content:
                if term == s:
                    j += 1
            inverted_index[term].append(j)
            print(inverted_index[term])
    return dict(inverted_index)


# print('corpus ', corpus)
# print('-------------------------------------------------------------------------------------------------')
# print(
#     '------------------------------------------------ inverted index -------------------------------------------------')
# print('-------------------------------------------------------------------------------------------------')
#
# inverted_index = create_inverted_index(corpus)
# print(
#     '----------------------------------------------- end inverted index-----------------------------------------------')

def create_tf_idf(corpus):
    documents = list(corpus.values())
    documents_as_strings = [' '.join(doc) for doc in documents]
    print('docs : ', documents_as_strings)

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    print('BB', vectorizer)

    # Fit the vectorizer to the documents
    tfidf_matrix = vectorizer.fit_transform(documents_as_strings)

    # Convert the TF-IDF matrix to a Pandas DataFrame  this is the inverted index
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=corpus.keys())

    # Print the resulting TF-IDF scores
    print(df)
    return vectorizer, tfidf_matrix


# ----------------------------------------fourth request----------------------------------

def cos_similarity(search_query_weights, tfidf_weights_matrix):
    cosine_distance = cosine_similarity(search_query_weights, tfidf_weights_matrix)
    similarity_list = cosine_distance[0]
    print('cod ', cosine_distance)

    return similarity_list


def most_similar(similarity_list, min_Doc=10):
    most_similar = []

    tmp_index = np.argmax(similarity_list)
    while tmp_index != 0:
        tmp_index = np.argmax(similarity_list)
        print('max ', np.argmax(similarity_list))
        if similarity_list[tmp_index] < 0.2:
            break
        print('similiraty list ', len(similarity_list), similarity_list)
        print('tmp index', tmp_index)
        if tmp_index != 0:
            most_similar.append(tmp_index)
        similarity_list[tmp_index] = 0

        # min_Doc -= 1
    return most_similar


def calculate_map(queries, dataset):
    map_scores = []
    corpus, raw_docs = process_dataset(dataset)
    vectorizer, tfidf_matrix = create_tf_idf(corpus)
    for i in range(len(queries)):
        query = queries[i][1]
        print('query ', query)

        docs = search_for_query(corpus, raw_docs, vectorizer, tfidf_matrix, query, 0)

        with open("antique-test.txt", "r") as qrel:
            relevant_docs = []
            for l in qrel:
                res = l.split()
                if str(res[0]) == str(queries[i][0]):
                    relevant_docs.append(res[2])
        print('relevant docs ', relevant_docs)
        num_relevant_docs = len(relevant_docs)
        num_retrieved_docs = 0
        num_correct_docs = 0
        precision_at_k = []
        recall_at_k = []
        print(' num relevant docs ', relevant_docs)

        for doc in docs:
            if doc in relevant_docs:
                num_correct_docs += 1
            num_retrieved_docs += 1
            precision = num_correct_docs / num_retrieved_docs
            if num_relevant_docs > 0:
                recall = num_correct_docs / num_relevant_docs
            else:
                recall = 0
            precision_at_k.append(precision)
            recall_at_k.append(recall)

        ap = 0
        for rec in relevant_docs:
            if rec in docs:
                k = docs.index(rec)
                print('k ', k)
                ap += precision_at_k[k]
        if num_relevant_docs > 0:
            print('ap ', ap, num_relevant_docs)
            ap /= num_relevant_docs
        map_scores.append(ap)
    print('score ', map_scores)

    # compute mean average precision
    map_score = np.mean(map_scores)
    return map_score

@app.get("/search")
def search(query: str, dataset: int):
    name = unicodedata.name(query[0]).lower()
    if 'arabic' in name:
        query = GoogleTranslator(source='auto', target='en').translate(query)
        print('query after translate', query)
    spell = Speller(lang='en')

    # query = spell(query)
    tokens = word_tokenize(query)
    tokens = [w.lower() for w in tokens]
    # filter out stop words
    words = [w for w in tokens if not w in stop_words]
    words = [w for w in words if not w in stop_chars]

    print('stem words in query', list(words))

    qpos_tagged = nltk.pos_tag(words)
    qwordnet_tagged = list(map(lambda x: (x[0], part_of_speech_tagger(x[1])), qpos_tagged))
    lemmatized_sentence = []
    for word, tag in qwordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    query_as_strings = ' '.join(lemmatized_sentence)
    print('query ', query_as_strings)
    corpus, raw_docs = process_dataset(dataset)
    vectorizer, tfidf_matrix = create_tf_idf(corpus)
    query_tfidf_matrix = vectorizer.transform([query_as_strings])
    kmeans = KMeans(n_clusters=5, random_state=0).fit(tfidf_matrix)
    cluster_assignments = kmeans.labels_
    print('query_tfidf_matrix ', query_tfidf_matrix)
    print('tfidf_matrix ', tfidf_matrix)
    r = cos_similarity(query_tfidf_matrix, tfidf_matrix)
    docs = []
    most_similar_data = most_similar(r)

    length = len(most_similar_data)
    print(' length ', length)
    for k in range(length):
        docs.append(list(raw_docs.values())[most_similar_data[k]])
    return {"most_similar_docs": docs}


def search_for_query(corpus, raw_docs, vectorizer , tfidf_matrix ,query: str, dataset: int):

    tokens = word_tokenize(query)
    tokens = [w.lower() for w in tokens]
    # filter out stop words
    words = [w for w in tokens if not w in stop_words]
    words = [w for w in words if not w in stop_chars]

    print('stem words in query', list(words))

    qpos_tagged = nltk.pos_tag(words)
    qwordnet_tagged = list(map(lambda x: (x[0], part_of_speech_tagger(x[1])), qpos_tagged))
    lemmatized_sentence = []
    for word, tag in qwordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    query_as_strings = ' '.join(lemmatized_sentence)
    print('query ', query_as_strings)
    query_tfidf_matrix = vectorizer.transform([query_as_strings])
    kmeans = KMeans(n_clusters=5, random_state=0).fit(tfidf_matrix)
    cluster_assignments = kmeans.labels_
    print('query_tfidf_matrix ', query_tfidf_matrix)
    print('tfidf_matrix ', tfidf_matrix)
    r = cos_similarity(query_tfidf_matrix, tfidf_matrix)
    docs = []
    most_similar_data = most_similar(r)

    length = len(most_similar_data)
    print(' length ', length)
    for k in range(length):
        docs.append(list(raw_docs.values())[most_similar_data[k]])
    return {"most_similar_docs": docs}


def apply_clustering(raw_docs, cluster_assignments, r, kmeans, vectorizer, query_tfidf_matrix):

    doc_ids = list(raw_docs.keys())

    all_doc_cluster_assignments = cluster_assignments[np.argsort(r)]

    # compute the cluster assignments of the top-ranked documents
    num_top_docs = 10
    top_doc_indices = np.argsort(r)[-num_top_docs:]
    top_doc_cluster_assignments = all_doc_cluster_assignments[top_doc_indices]

    # use the cluster assignments to re-rank the top-ranked documents
    cluster_weights = np.bincount(top_doc_cluster_assignments, minlength=kmeans.n_clusters)
    cluster_weights = cluster_weights / np.sum(cluster_weights)
    cluster_sim = np.zeros(kmeans.n_clusters)

    for i in range(kmeans.n_clusters):
        cluster_doc_ids = [doc_ids[j] for j in range(len(doc_ids)) if all_doc_cluster_assignments[j] == i]
        if len(cluster_doc_ids) > 0:
            cluster_docs = [raw_docs[doc_id] for doc_id in cluster_doc_ids]
            cluster_tfidf_matrix = vectorizer.transform(cluster_docs)
            cluster_sim[i] = cosine_similarity(query_tfidf_matrix, cluster_tfidf_matrix).mean()

    # rank the clusters by their weighted average similarity scores
    re_ranked_indices = np.argsort(-cluster_weights * cluster_sim)[:num_top_docs]

    # retrieve the most similar documents
    docs = []
    for i in re_ranked_indices:
        cluster_doc_ids = [doc_ids[j] for j in range(len(doc_ids)) if
                           all_doc_cluster_assignments[j] == re_ranked_indices[i]]
        if len(cluster_doc_ids) > 0:
            cluster_docs = [raw_docs[doc_id] for doc_id in cluster_doc_ids]
            docs.append(
                cluster_docs[np.argmax(cosine_similarity(query_tfidf_matrix, vectorizer.transform(cluster_docs)))])
    return docs


def calculate_recall(query_id, relevant_docs_retrieved):
    with open("antique-test.txt", "r") as qrel:
        relevant_docs_in_corups = 0
        for l in qrel:
            res = l.split()
            if str(res[0]) == str(query_id):
                relevant_docs_in_corups += 1
    if relevant_docs_in_corups > 0:
        recall = relevant_docs_retrieved / relevant_docs_in_corups
    else:
        return 0
    return recall

@app.get("/evaluate")
def evaluate_results(dataset: int):
    with open("queries.txt", "r") as file:
        data = {}
        z = 0
        recall_sum = 0
        queries_count = 0
        queries = []
        corpus, raw_docs = process_dataset(dataset)
        vectorizer, tfidf_matrix = create_tf_idf(corpus)
        for line in file:
            queries_count += 1
            query_id = line[0:7]
            query = line[7:]
            queries.append([query_id, query])
            print('id ', query_id)
            print('query ', query)
            tokens = word_tokenize(query)
            tokens = [w.lower() for w in tokens]
            words = [w for w in tokens if not w in stop_words]
            words = [w for w in words if not w in stop_chars]

            qpos_tagged = nltk.pos_tag(words)
            qwordnet_tagged = list(map(lambda x: (x[0], part_of_speech_tagger(x[1])), qpos_tagged))
            print("wordnet_tagged :", qwordnet_tagged)

            lemmatized_sentence = []
            for word, tag in qwordnet_tagged:
                if tag is None:
                    lemmatized_sentence.append(word)
                else:
                    print('q before lemmatize ', word)
                    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
                    print('after ', lemmatizer.lemmatize(word, tag), tag, word)

            query_as_strings = ' '.join(lemmatized_sentence)
            print('query ', query_as_strings)
            query_tfidf_matrix = vectorizer.transform([query_as_strings])
            r = cos_similarity(query_tfidf_matrix, tfidf_matrix)
            print('cosine ', r)
            # print('most_similar ', list(corpus.values())[most_similar(r)[1]])
            docs = []
            most_similar_data = most_similar(r)
            length = len(most_similar_data)
            recall_sum += calculate_recall(query_id, length)
            z += length
            for k in range(length):
                docs.append(list(raw_docs.values())[most_similar_data[k]])
            data[query_id] = docs
            print('docs ', docs)
            print('z ', z)
        final_recall = recall_sum / queries_count
        print('MAP ', calculate_map(queries, dataset))
        print('final recall ', final_recall)
        return data

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import triu


def cosine_similarity_method(doc_bow, threshold):
    similarity = triu(cosine_similarity(doc_bow, dense_output=False), k=1)
    coincidences = np.argwhere(similarity > threshold)
    delete_indices = []
    n_words = doc_bow.sum(axis=1)
    for coincidence in coincidences:
        smallest_doc = np.argmin([n_words[coincidence[0]].item(), n_words[coincidence[1]].item()])
        delete_indices.append(coincidence[smallest_doc])

    delete_indices_unique = np.unique(delete_indices)
    all_indices = np.arange(doc_bow.shape[0])
    return np.setdiff1d(np.union1d(delete_indices_unique, all_indices), np.intersect1d(delete_indices_unique, all_indices))


def unique_rows_csr(doc_bow_csr, similarity_threshold=1):
    keep_indices = cosine_similarity_method(doc_bow_csr, similarity_threshold)
    return doc_bow_csr[keep_indices.astype(int)]
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

import pickle
import numpy as np

data = pickle.loads(open("output/embeddings.pickle", "rb").read())
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())
embed_faces = data["embeddings"]
y_labels = data["ids"]
ids = np.arange(len(y_labels))

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(np.stack(embed_faces), y_labels, ids,
                                                                       test_size=0.2, stratify=y_labels)
print(X_train.shape, X_test.shape)
print(len(y_train), len(y_test))


def _most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    sim = np.squeeze(sim, axis=1)
    argmax = np.argsort(sim)[::-1][:1]
    label = [labels[idx] for idx in argmax][0]
    return label


y_predsMS = []
y_predsSVM = []
for vec in X_test:
    vec = vec.reshape(1, -1)

    y_pred_ms = _most_similarity(X_train, vec, y_train)

    predicts = recognizer.predict_proba(vec)[0]
    j = np.argmax(predicts)
    y_pred_svm = le.classes_[j]

    y_predsMS.append(y_pred_ms)
    y_predsSVM.append(y_pred_svm)

print("Most similarity Accuracy: " + str(accuracy_score(y_predsMS, y_test)))
print("SVM Accuracy: " + str(accuracy_score(y_predsSVM, y_test)))
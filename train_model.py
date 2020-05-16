from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


def run():
    print("[LOG] Loading Face Embeddings")
    data = pickle.loads(open("output/embeddings.pickle", "rb").read())

    print("[LOG] Encoding Labels")
    le = LabelEncoder()
    labels = le.fit_transform(data["ids"])

    print("[LOG] Training Model")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # Write the face recognition model to disk
    f = open("output/recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # Write the label encoder to disk
    f = open("output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

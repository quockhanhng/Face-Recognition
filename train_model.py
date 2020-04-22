from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True, help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
args = vars(ap.parse_args())

print("[LOG] Loading Face Embeddings")
data = pickle.loads(open(args["embeddings"], "rb").read())

print("[LOG] Encoding Labels")
le = LabelEncoder()
labels = le.fit_transform(data["ids"])

print("[LOG] Training Model")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Write the face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

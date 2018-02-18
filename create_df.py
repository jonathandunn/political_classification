#This script creates feature vectors for input texts, with meta-data if available

vectorizer_name = "Vectorizer.RAW.eng.LEX.100k.Legislative.p"
input_files = ["Corpus.US.Senate.1995-2012.txt"]
class_name = "Party"		#This dictates which class is used
language = "eng"
pos = False

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack
from scipy.sparse import save_npz
from modules.Encoder import Encoder
from modules.Loader import Loader

#-- Input from encoder already prepared
def preprocessor(line):
	return [x for x in line]

#Set input and output paths; reads all .txt files in the specified input directory
in_dir = os.path.join("..", "..", "Data", "Legislative")
out_dir = os.path.join("..", "..", "Data")
Load = Loader(in_dir, out_dir)	

#Get speaker meta-data; only covers data from US House and Senate
speakers = Load.load_speakers()

#Initiate Loader; all files in input_directory that end with ".txt" will be used
Encode = Encoder(language = language, Loader = Load)

#Texts is a generator that produces streams of documents
Texts = Encode.load_stream(input_files, pos = pos, enrich = True)		#If no meta-data, 'enrich = False'

with open(os.path.join(".", "data", vectorizer_name), "rb") as handle:
	Vectorizer = pickle.load(handle)
	
feature_list = []	#Hold feature arrays (sparse numpy arrays)
meta_list = []		#Hold meta-data (Python list)
	
#For each line, get text features and meta-data
for line in Texts:
	
	text_id = line[0]
	speaker_id = line[1]
	line = line[2]
	features = Vectorizer.transform(line)
	
	try:
		meta_data = speakers[speaker_id][class_name]
		feature_list.append(features)
		meta_list.append(meta_data)
		
	except:
		print("Missing meta-data for " + str(speaker_id))
	
#Now merge into dataframe
features = vstack(feature_list)
meta = np.array(meta_list)

filename = "Senate." + vectorizer_name + ".Features"
save_npz(os.path.join(in_dir, filename), features, compressed = True)

filename = "Senate." + vectorizer_name + ".Classes"
np.save(os.path.join(in_dir, filename), meta, allow_pickle = True)
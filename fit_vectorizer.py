#This script trains new CountVectorizer given a folder of .txt files

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from modules.Encoder import Encoder
from modules.Loader import Loader

language = "eng"
pos = False

#-- Input from encoder already prepared
def preprocessor(line):
	return [x for x in line]
#--------------------------------------	
	
#Set input and output paths; reads all .txt files in the specified input directory
in_dir = os.path.join("..", "..", "Data", "Background")
out_dir = os.path.join("..", "..", "Data")
Load = Loader(in_dir, out_dir)	

#Initiate Loader; all files in input_directory that end with ".txt" will be used
Encode = Encoder(language = language, Loader = Load)

#Texts is a generator that produces streams of documents
input_files = Load.list_input()
Texts = Encode.load_stream(input_files, pos = pos)

Vectorizer = TfidfVectorizer(input = "content", 
				encoding = "utf-8", 
				decode_error = "replace",
				strip_accents = None, 
				lowercase = False, 
				stop_words = None, 
				ngram_range = (1, 1), 
				analyzer = preprocessor,
				min_df = 1, 
				max_features = 100000
				)
				
Vectorizer.fit([x for x in Texts])
features = Vectorizer.get_feature_names()

RawVectorizer = CountVectorizer(input = "content", 
				encoding = "utf-8", 
				decode_error = "replace",
				strip_accents = None, 
				lowercase = False, 
				stop_words = None, 
				ngram_range = (1, 1), 
				analyzer = preprocessor,
				min_df = 1, 
				max_features = 100000,
				vocabulary = features
				)

#-- Get correct name
if pos == True:
	pos_name = "POS"
else:
	pos_name = "LEX"
#------------------

filename = "Vectorizer.TFIDF." + language + "." + pos_name + ".100k.Legislative.p"
Load.save_file(Vectorizer, filename)

filename = "Vectorizer.RAW." + language + "." + pos_name + ".100k.Legislative.p"
Load.save_file(RawVectorizer, filename)

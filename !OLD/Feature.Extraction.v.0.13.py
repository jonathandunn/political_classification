#Script for creating feature vectors for several frequency features from input files#
#Jonathan Dunn, Geet Kumar, March - October, 2014#
#Linguistic Cognition Lab, Illinois Institute of Technology#

#Current Features Produced:# 	
	#SpeechLengthAll#
	#SpeechLengthLexical#
	#TypeTokenAll#
	#TypeTokenLexical#
	#AvgWordLengthAll#
	#AvgWordLengthLexical#
	#StdWordLengthAll#
	#StdWordLengthLexical#
	#Relative frequency for each word in each document (where word appears in at least N number of documents)#
	#TF-IDF for each word in each document (where word appears in at least N number of documents)#
	#Difference between relative frequency in document and baseline relative frequency (where word appears in at least N number of documents)#
	
#Current Representations:#
	#N-Grams#
		#Word Forms#
		
	

	

	
#-----------------------------------------------------------------------------------------------------#
#STEP 1: IMPORT DEPENDENCIES#

import re									 		#Import Regular Expression module#
import numpy								 		#Import NumPy module#
#import nltk									  		#Import Natural Language Toolkit modules##
#from nltk.stem.wordnet import WordNetLemmatizer		#Import the WordNet lemmatizer##
import codecs
#from stat_parser import Parser                     #Import parser; https://github.com/emilmont/pyStatParser#

#END OF STEP 1: IMPORT DEPENDENCIES#
#-----------------------------------------------------------------------------------------------------#




#-----------------------------------------------------------------------------------------------------#
#STEP 2: DEFINE CONSTANTS#

input_files = [                              				 #List and path of files in the dataset#
			'104-109.Individual.Token.POS.txt'
						 ]
						 
input_files_aggregated = [                              				 #List and path of files in the dataset#
			'104-109.Aggregated.Token.POS.txt'
						 ]						 
						 
			 
output_feature_type = 'RELATIVE'				 #Determine which type of frequency value to use: 'TF-IDF', 'RELATIVE', 'RF-BRF', or 'ALL'#
n_gram_number = [1,2,3]						 #Set the number N for the n-gram features; unigrams are '1'#
representation_type = 'WORDFORM'				 #Determine which type of representation is used: 'WORDFORM', 'LEMMA', or 'POS'#
frequency_threshold = 0.025						 #Minimum number of documents word or n-gram must occur in before it is included as a feature#


#Material added to input file name for corresponding output file#

output_suffix = 'Word-Form+PoS.1-3.RelativeFreq.Threshold.02.arff'	 
output_suffix_aggregated = 'Word-Form+PoS.1-3.RelativeFreq.Threshold.02.arff'

speaker_properties_file = 'Speaker.Index.Scale.txt'				 #Location and name for speaker properties#

stopwords = [
'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t',
'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have',
'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers',
'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve',
'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more',
'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only',
'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t',
'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than',
'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s',
'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to',
'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve',
'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while',
'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d',
'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
]



interest_groups = [
'1:Institutions','2:Individuals','3:Animals'
]

#END OF STEP 2: DEFINE CONSTANTS#
#-----------------------------------------------------------------------------------------------------#




#-----------------------------------------------------------------------------------------------------#
#STEP 3: DEFINE FUNCTIONS#
#-----------------------------------------------------------------------------------------------------#
#
#--------------------------------LIST OF FUNCTIONS----------------------------------------------------#

#A. INPUT FUNCTIONS-----------------------------------------------------------------------------------#


#A1#     get_document_id(line)       	This function takes a line from the Congressional Record data and returns the speech id#
#A2#     get_congress_id(line)       	This function takes a line from the Congressional Record data and returns the congress id#
#A3#     get_speaker_id(line)        	This function takes a line from the Congressional Record data and returns the speaker id#
#A4#     get_document_text(line,flag)   This function takes a line from the Congressional Record data and returns the speech as a string. The flag variable indicates the level of representation to be used.#
#A5#	 get_document_chamber(line)		This function takes a line from the Congressional Record data and returns the chamber that speech was given in.#


#B. TEXT PROCESSING FUNCTIONS-------------------------------------------------------------------------#

	
#B1#	tokenize_line(line,flag)						This function takes a line from the Congressional Record data and returns a tokenized version of the speech, as a list of words. The flag variable indicates the level of representation to be used.#
#B2#	extract_n_gram(document_text,n_gram_number)		This function takes a list of words in the text and the N for n-grams and returns a list of n-grams in the document, including duplicates#
#B3#	split_sentences(document_text)					This function takes a list of words in the text and returns the text split into sentences#
#B4#	get_lemmas(document_text)						This function takes a sentence as a list of words and returns a list of lemmas maintains sequential order without removing any duplicate lemmas#
#B5#	get_pos(document_text)							This function takes a sentence as a list of words and returns a list of part of speech tags
#B6#	parse_sentences(document_text)					Kumar. This function takes a string of text, splits it into sentences, and returns a list of parsed sentences


#C. FEATURE PREPARATION FUNCTIONS---------------------------------------------------------------------#


#C1#	count_documents(input_files)																				This function takes a list of input files containing Congressional Record data and returns the number of speech which they contain#	
#C2#	create_n_gram_list(input_files,n_gram_number)																This function takes the list of input files and the N for n-grams and returns a dictionary with all word n-grams thus defined and the number of documents they occur in.#
#C3#	trim_n_gram_list(n_gram_list,threshold)																		This function takes a dictionary of word/number of documents containing pairs and returns a dictionary of words occuring in over a set number of documents (defined in the threshold input variable)
#C4#	create_n_gram_list_keys(n_gram_list)																		This function takes a dictionary of words with frequency values and returns a sorted list of the keys#
#C5#	count_total_n_grams(input_files)																			This function counts the total number of word instances in the documents.#
#C6#	count_list_n_grams(input_files,n_gram_list_keys,n_gram_number)												This function counts the total frequency of ngrams on the ngram list in all the documents.#	
#C7#	calculate_inverse_document_frequency(n_gram_list,number_of_documents)										This function taks a dictionary with words as entries and the number of documents containing the words as values along with the total number of documents in the data set and returns a dictionary with words as entries and inverse document frequency for each word as values#
#C8#	calculate_total_relative_frequency(n_gram_list_frequency,total_words,n_gram_list_keys,n_gram_number)		This function counts returns the relative frequency for each word on the word list in all documents.#	


#D. FEATURE CALCULATION FUNCTIONS--------------------------------------------------------------------#


#D1#	 type_token_all(document_text)																This function takes a tokenized speech text and returns the type / token ratio for all words#
#D2#	 type_token_lexical(document_text,stopwords)												This function takes a tokenized speech text and returns the type / token ratio for only non-stopwords#
#D3#	 avg_word_length_all(document_text)															This function takes a tokenized speech text and returns the average length in characters of all words#
#D4#	 avg_word_length_lexical(document_text,stopwords)											This function takes a tokenized speech text and returns the average length in characters of non-stopwords#
#D5#	 std_word_length_all(document_text)															This function takes a tokenized speech text and returns the standard deviation in characters of word length for all words#
#D6#	 std_word_length_lexical(document_text,stopwords)											This function takes a tokenized speech text and returns the standard deviation in characters of word length for non-stopwords#
#D7#	 calculate_relative_frequency(document_text,word,n_gram_number)								This function takes a tokenized speech text and a word and returns the relative frequency of that word in that speech#
#D8#	 calculate_tfidf_frequency(document_text,word,n_gram_list,n_gram_number)					This function takes a tokenized speech text, a word, and a dictionary of inverted document frequencies and returns the term-frequency(inverted document frequency) for that word in that speech#	
#D8#	 calculate_rfbrf_frequency(document_text,word,n_gram_list_frequency,n_gram_number)			This function takes a tokenized speech text, a word, and a dictionary of total relative word frequencies and returns the relative frequency - baseline relative frequency for that word in that speech#	


#E. DOCUMENT ATTRIBUTE FUNCTIONS----------------------------------------------------------------------#


#E1#	create_speaker_list(input_files)													This function takes a list of input files containing Congressional Record data and returns a list of speakers contained in those files#
#E2#	create_speaker_dictionary(speaker_properties_file,speaker_list,interest_groups)		This function takes as input a file containing speaker properties and a list of speakers present in the dataset and returns a dictionary of dictionaries with the necessary speaker properties#


#F. OUTPUT FUNCTIONS----------------------------------------------------------------------------------#


#F1#	create_output_file_names(input_files,output_suffix)																							This function takes a list of input files and a specified suffix for output files and returns a list of output files corresponding to each#
#F2#	write_arff_headers(output_files,speaker_list,n_gram_list_keys,interest_groups)																This function writes the ARFF headers for each output file, taking as input the set of output files and a list of speakers, a list of words, and a list of interest groups#
#F3#	write_frequency_vector(line,n_gram_list,n_gram_list_frequency,n_gram_list_keys,speaker_properties,interest_groups,stopwords,n_gram_number)	This function takes a line from the Congressional Record data and writes its vector.#


#--------------------------------END LIST OF FUNCTIONS------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
#-----------------------------BEGIN FUNCTION DEFINITIONS----------------------------------------------#


#------------------------------------------------------------------------------------------------------#
#A. INPUT FUNCTIONS-----------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#


#A1 get_document_id--------------------------------------------------------------------------------------#
#Dunn. This function takes a line from the Congressional Record data and returns the speech id#
def get_document_id(line):
	temp_list = line.split('\t')
	speech_id = temp_list[0]					 #Return speech id for current document#
	return speech_id
#------------------------------------------------------------------------------------------------------#


#A2 GET_CONGRESS_ID------------------------------------------------------------------------------------#
#Dunn. This function takes a line from the Congressional Record data and returns the congress id#
def get_congress_id(line):
	temp_list = line.split('\t')
	congress_id = temp_list[1]			 		 #Return congress id for current document#
	return congress_id
#------------------------------------------------------------------------------------------------------#


#A3 GET_SPEAKER_ID-------------------------------------------------------------------------------------#
#Dunn. This function takes a line from the Congressional Record data and returns the speaker id#
def get_speaker_id(line):
	temp_list = line.split('\t')
	speaker_id = temp_list[0]
	return speaker_id
#------------------------------------------------------------------------------------------------------#	


#A4 get_document_text------------------------------------------------------------------------------------#
#Dunn. This function takes a line from the Congressional Record data and returns the speech as a string.#
def get_document_text(line,flag):

	temp_list = line.split('\t')
	document_text = temp_list[3]
	
	document_text = document_text.lower()
	
	return document_text
#-------------------------------------------------------------------------------------------------------#


#A5 get_document_chamber------------------------------------------------------------------------------------#
#Dunn. This function takes a line from the Congressional Record data and returns the chamber in which it was given.#
def get_document_chamber(line):

	temp_list = line.split('\t')
	chamber_id = temp_list[2]
	
	return chamber_id				#Return first character, which is the chamber#
#-------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------#
#B. TEXT PROCESSING FUNCTIONS-------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#


#B1 TOKENIZE_LINE---------------------------------------------------------------------------------------#
#Dunn. This function takes a line from the Congressional Record data and returns a tokenized version of the speech, as a list of words#	
def tokenize_line(line,flag):
	
	#CONVERT DOCUMENT TO LIST OF WORDS#				
	temp_list = get_document_text(line, flag)

	
	temp_list = temp_list.split()           			#Convert string into list of words
	
	temp_list_2 = []

	if flag == 'WORDFORM':
		
		temp_list_2 = temp_list
	
	elif flag == 'POS':
	
		for word in temp_list:
			tag_begin = word.find('_')
			temp_word = word[tag_begin+1:]
			temp_list_2.append(temp_word)
		

	
		
	return temp_list_2
#---------------------------------------------------------------------------------------------------------#


#B2 EXTRACT_N_GRAM--------------------------------------------------------------------------------------#
#Dunn. This function takes a list of words in the text and the N for n-grams and returns#
#A list of n-grams in the document, including duplicates.#
def extract_n_gram(document_text,n_gram_number):	
	
	temp_n_gram_list = []
	
	for g in n_gram_number:
	
		n_gram_number_temp = g
	
		for i in range(len(document_text)):	#Go through each word, looking forward for n-grams#
		
			if (i + (n_gram_number_temp - 1)) <= len(document_text):				#Make sure that N words forward is within the text#
				temp = document_text[i:i + (n_gram_number_temp)]					#Define n-gram window, moving forward only#
										
				#Start to turn list of words into single n-gram string#
				n_gram_text = ''			
					
				for z in range(len(temp)):
					if z == 0:
						n_gram_text = temp[z]
					elif z > 0:
						n_gram_text += '.' + temp[z]
				#End turn list of words into single n-gram string#
					
				temp_n_gram_list += [n_gram_text]			#Add current n-gram to n-gram list for this document#
					
	return temp_n_gram_list
#------------------------------------------------------------------------------------------------------#


#B3 SPLIT_INTO_SENTENCES-------------------------------------------------------------------------------#
#Kumar. This function takes a list of words in the text and returns a list split into sentences.
def split_into_sentences(text):
 
	sent_detector = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
 
	sentences = sent_detector.tokenize(text.strip())
	
	return sentences
#------------------------------------------------------------------------------------------------------#


#B4 GET_LEMMAS-----------------------------------------------------------------------------------------#
#Kumar. This function takes a sentence as a list of words and returns a list of lemmas maintains sequential order without removing any duplicate lemmas
def get_lemmas(wordLst):

    lemmatizer = WordNetLemmatizer()
    result = []
    for word in wordLst:
        result.append(lemmatizer.lemmatize(word.lower()))
		
    return result
#------------------------------------------------------------------------------------------------------#


#B5 GET_POS-----------------------------------------------------------------------------------------#
#Kumar. This function takes a sentence as a list of words and returns a list of part of speech tags
def get_pos(wordLst):
    
    return [tpl[1] for tpl in nltk.pos_tag(wordLst)]
#------------------------------------------------------------------------------------------------------#


#B6 PARSE_SENTENCES-----------------------------------------------------------------------------------------#
#Kumar. This function takes a string of text, splits it into sentences, and returns a list of parsed sentences
def parse_sentences(txt):
    
    parser=Parser()
    result = []
    sentences = split_into_sentences(txt)
    for sent in sentences:
        result.append(str(parser.parse(sent)))
    
    return result
#------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------#
#C. FEATURE PREPARATION FUNCTIONS---------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#


#C1 COUNT_DOCUMENTS----------------------------------------------------------------------------------------#
#Dunn. This function takes a list of input files containing Congressional Record data and returns the number of speech which they contain#	
def count_documents(input_files):
	
	speech_count = 0					#Initiate speech count#
	
	for file in input_files:			#Loop through all files#
		fo = codecs.open(file, 'rb', 'utf-8')
		
		for line in fo:					#Loop through all lines in file#
			speech_count += 1
		
		fo.close()						#Close each file after looping through all documents#
	
	return speech_count
#------------------------------------------------------------------------------------------------------#


#C2 CREATE_N_GRAM_LIST-----------------------------------------------------------------------------#
#Dunn. This function takes the list of input files and the N for n-grams and returns#
#A dictionary with all word n-grams thus defined and the number of documents they occur in.#
def create_n_gram_list(input_files,n_gram_number,flag,threshold,number_of_documents):

	text_n_gram_dictionary = {}
	temp_n_gram_dictionary = {}
	
	for g in n_gram_number:			#Loop through each numbe rin n-gram range#
	
		current_n_gram = []
		current_n_gram.append(g)
	
		for file in input_files:			#Loop through all files#
			fo = codecs.open(file, 'rb', 'utf-8')
		
			for line in fo:					#Loop through all lines in file#
				text = tokenize_line(line,flag)	#Tokenize the line and remove meta-data#
			
			
				temp_n_gram_list = extract_n_gram(text,current_n_gram)
				temp_n_gram_list = set(temp_n_gram_list)
			
				for n_gram in temp_n_gram_list:						#Loop through n-grams from current document and add to overall list#
			
					if n_gram in temp_n_gram_dictionary:							
						temp_n_gram_dictionary[n_gram] = temp_n_gram_dictionary[n_gram] + 1	#If word n-gram already in word list, increase frequency by 1#
					
					else:
						temp_n_gram_dictionary[n_gram] = 1			               		#If word n-gram not in word list, add with frequency of 1# 
			
			print('Done with ' + file + ' N-Gram: ' + str(g))
			fo.close()						#Close each file after looping through all documents#
		
		deletion_list = []									#Initiate list of rare words to be deleted#
			
		for entry, documents in temp_n_gram_dictionary.items():		#Loop through words and the number of documents they occur in#
			if documents <= (float(threshold) * number_of_documents):
				deletion_list.append(entry)					#Create list of words to be removed#
			
		for entry in deletion_list:
			del temp_n_gram_dictionary[entry]							#Remove words on deletion list#
				
		text_n_gram_dictionary.update(temp_n_gram_dictionary)
			
	return text_n_gram_dictionary	
#------------------------------------------------------------------------------------------------------#

#C4 CREATE_N_GRAM_LIST_KEYS------------------------------------------------------------------------------#
#Dunn. This function takes a dictionary of words with frequency values and returns a sorted list of the keys#
def create_n_gram_list_keys(n_gram_list):
	
	n_gram_list_keys = n_gram_list.keys()             		  #Create list of all words in documents, now that words found in few documents have been removed#
	n_gram_list_keys = sorted(set(n_gram_list_keys))		  #Sort word list#
	return n_gram_list_keys
#------------------------------------------------------------------------------------------------------#


#C5 COUNT_TOTAL_N_GRAMS----------------------------------------------------------------------------------#
#Dunn. This function counts the total number of word instances in the documents.#
def count_total_n_grams(input_files):
	
	total_words = 0						#Initiate counter for total words in documents#
	
	for file in input_files:			#Loop through all files#
		fo = codecs.open(file, 'rb', 'utf-8')
		
		for line in fo:					#Loop through all lines in file#
			text = tokenize_line(line,'none')	#Tokenize the line and remove meta-data#
			total_words += len(text)	#Add number of words in current document to total word counter#
		
		print('Done with ' + file)
		fo.close()						#Close each file after looping through all documents#
		
	return total_words
#------------------------------------------------------------------------------------------------------#	


#C6 COUNT_LIST_N_GRAMS----------------------------------------------------------------------------------#
#Dunn. This function counts the total frequency of ngrams on the ngram list in all the documents.#	
def count_list_n_grams(input_files,n_gram_list_keys,n_gram_number):

	n_gram_list_frequency = {}			#Initiate dictionary containing frequency of words on word list#
	
	for file in input_files:			#Loop through all files#
		fo = codecs.open(file, 'rb', 'utf-8')
		
		for line in fo:					#Loop through all lines in file#
			text = tokenize_line(line,'none')	#Tokenize the line and remove meta-data#
			temp_n_gram_list = extract_n_gram(text,n_gram_number)	#Find list of n-grams in text#
			
			for word in temp_n_gram_list:
				if word in n_gram_list_keys:				#Loop through words, check if word is on word list#
										
					if n_gram_list_frequency.has_key(word):							
						n_gram_list_frequency[word] = n_gram_list_frequency[word] + 1	#If word already in word list, increase frequency by 1#
					
					else:
						n_gram_list_frequency[word] = 1			               		#If word not in word list, add with frequency of 1# 
		
		print('Done with ' + file)
		fo.close()						#Close each file after looping through all documents#
		
	return n_gram_list_frequency
#------------------------------------------------------------------------------------------------------#


#C7 CALCULATE_INVERSE_DOCUMENT_FREQUENCY---------------------------------------------------------------#
#Dunn. This function taks a dictionary with words as entries and the number of documents containing the words#
#as values along with the total number of documents in the data set and returns a dictionary#
#with words as entries and inverse document frequency for each word as values#
def calculate_inverse_document_frequency(n_gram_list,number_of_documents):
	
	for entry, documents in n_gram_list.items():
	
		number_of_documents_containing = n_gram_list[entry]											#Find number of documents containing the current word#
		inverse_document_frequency = float(number_of_documents) / number_of_documents_containing	#Find ratio of total documents / documents containing the current word#
		n_gram_list[entry] = numpy.log(inverse_document_frequency)									#Find and store logarithm of the ratio#

	return n_gram_list
#------------------------------------------------------------------------------------------------------#

	
#C8 CALCULATE_TOTAL_RELATIVE_FREQUENCY----------------------------------------------------------------#
#Dunn. This function counts returns the relative frequency for each word on the word list in all documents.#	
def calculate_total_relative_frequency(n_gram_list_frequency,total_words,n_gram_list_keys,n_gram_number):
	
	for word in n_gram_list_keys:									#Loop through words on word list#
		total_freq = n_gram_list_frequency[word]					#Save total frequency of current word#
		n_gram_list_frequency[word] = float(total_freq) / total_words	#Set relative frequency for each word in all documents#
		
	return n_gram_list_frequency		
#------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------#
#D. FEATURE CALCULATION FUNCTIONS--------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#


#D1 TYPE_TOKEN_ALL-------------------------------------------------------------------------------------#
#Dunn. This function takes a tokenized speech text and returns the type / token ratio for all words#
def type_token_all(document_text):
	if document_text:															
		type_token_all = len(set(document_text)) / float(len(document_text))						#Calculate type/token ratio for all words#
	else: type_token_all = 0
	
	return type_token_all
#------------------------------------------------------------------------------------------------------#


#D2 TYPE_TOKEN_LEXICAL---------------------------------------------------------------------------------#
#Dunn. This function takes a tokenized speech text and returns the type / token ratio for only non-stopwords#
def type_token_lexical(document_text,stopwords):
	if document_text:
		temp_list_lexical = [w for w in document_text if w not in stopwords]					#Remove stopwords#
		type_token_lexical = len(set(temp_list_lexical)) / float(len(temp_list_lexical))	#Calculate type/token ratio for non-stopwords#
	
	else: type_token_lexical = 0
	
	return type_token_lexical
#------------------------------------------------------------------------------------------------------#	


#D3 AVG_WORD_LENGTH_ALL--------------------------------------------------------------------------------#
#Dunn. This function takes a tokenized speech text and returns the average length in characters of all words#
def avg_word_length_all(document_text):

	word_lengths = []
	
	if document_text:
		for i in range(len(document_text)):							#Loop through list of all words to create list of word lengths#
			word_lengths.append(len(document_text[i]))
		avg_word_length_all = numpy.mean(word_lengths)				#Find average of list of word lengths for all words#
	else:
		avg_word_length_all = 0
		
	return avg_word_length_all
#------------------------------------------------------------------------------------------------------#


#D4 AVG_WORD_LENGTH_LEXICAL-----------------------------------------------------------------------------#
#Dunn. This function takes a tokenized speech text and returns the average length in characters of non-stopwords#
def avg_word_length_lexical(document_text,stopwords):

	word_lengths = []
	temp_list_lexical = [w for w in document_text if w not in stopwords]
	
	if temp_list_lexical:
		for i in range(len(temp_list_lexical)):					#Loop through list of non-stopwords to create list of word lengths#
			word_lengths.append(len(temp_list_lexical[i]))
		avg_word_length_lexical = numpy.mean(word_lengths)		#Find average of list of word lengths for non-stopwords#
	else:
		avg_word_length_lexical = 0
		
	return avg_word_length_lexical
#------------------------------------------------------------------------------------------------------#

	
#D5 STD_WORD_LENGTH_ALL--------------------------------------------------------------------------------#
#Dunn. This function takes a tokenized speech text and returns the standard deviation in characters of word length for all words#
def std_word_length_all(document_text):

	word_lengths = []
	
	if document_text:
		for i in range(len(document_text)):							#Loop through list of all words to create list of word lengths#
			word_lengths.append(len(document_text[i]))
		std_word_length_all = numpy.std(word_lengths)				#Find standard deviation of list of word lengths for all words#
	else:
		std_word_length_all = 0
		
	return std_word_length_all
#------------------------------------------------------------------------------------------------------#


#D6 STD_WORD_LENGTH_LEXICAL----------------------------------------------------------------------------#
#Dunn. This function takes a tokenized speech text and returns the standard deviation in characters of word length for non-stopwords#
def std_word_length_lexical(document_text,stopwords):

	word_lengths = []
	temp_list_lexical = [w for w in document_text if w not in stopwords]
	
	if temp_list_lexical:
		for i in range(len(temp_list_lexical)):							#Loop through list of all words to create list of word lengths#
			word_lengths.append(len(temp_list_lexical[i]))
		std_word_length_lexical = numpy.std(word_lengths)				#Find standard deviation of list of word lengths for all words#
	else:
		std_word_length_lexical = 0
		
	return std_word_length_lexical
#------------------------------------------------------------------------------------------------------#


#D7 CALCULATE_RELATIVE_FREQUENCY-----------------------------------------------------------------------#
#Dunn. This function takes a tokenized speech text and a word and returns the relative frequency of that word in that speech#
def calculate_relative_frequency(document_text,word,n_gram_number):
	
	if document_text:
		relative_frequency = document_text.count(word) / float(len(document_text))		#Calculate relative word frequency#
	else:
		relative_frequency = 0
			
	return relative_frequency
#------------------------------------------------------------------------------------------------------#


#D8 CALCULATE_TFIDF_FREQUENCY--------------------------------------------------------------------------#
#Dunn. This function takes a tokenized speech text, a word, and a dictionary of inverted document frequencies#
#and returns the term-frequency(inverted document frequency) for that word in that speech#	
def calculate_tfidf_frequency(document_text,word,n_gram_list,n_gram_number):

	if document_text:
		tfidf_frequency = document_text.count(word) * float(n_gram_list[word])		#Calculate weighted inverse document frequency#
	else:
		tfidf_frequency = 0

	return tfidf_frequency
#------------------------------------------------------------------------------------------------------#


#D9 CALCULATE_RFBRF_FREQUENCY--------------------------------------------------------------------------#	
#Dunn. This function takes a tokenized speech text, a word, and a dictionary of total relative word frequencies#
#and returns the relative frequency - baseline relative frequency for that word in that speech#	
def calculate_rfbrf_frequency(document_text,word,n_gram_list_frequency,n_gram_number):

	if document_text:
		rfbrf_frequency = (document_text.count(word) / float(len(document_text))) - float(n_gram_list_frequency[word])		#Calculate relative frequency - baseline relative frequency#
	else:
		rfbrf_frequency = 0

	return rfbrf_frequency
#------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------#
#E. DOCUMENT ATTRIBUTE FUNCTIONS----------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#


#E1 CREATE_SPEAKER_LIST--------------------------------------------------------------------------------#
#Dunn. This function takes a list of input files containing Congressional Record data and#
#Returns a list of speakers contained in those files#
def create_speaker_list(input_files):

	speaker_list = []									#Initiate list of speakers#
	
	for file in input_files:							#Loop through all files#
		fo = codecs.open(file, 'rb', 'utf-8')
		
		for line in fo:									#Loop through all lines in file#
			speaker = get_speaker_id(line)
			if speaker not in speaker_list:
				speaker_list.append(speaker)
		
		fo.close()										#Close each file after looping through all documents#
	
	return speaker_list
#------------------------------------------------------------------------------------------------------#	


#E2 CREATE_SPEAKER_DICTIONARY--------------------------------------------------------------------------#
#Dunn. This function takes as input a file containing speaker properties and a list of speakers present in the dataset#
#and returns a dictionary of dictionaries with the necessary speaker properties#
def create_speaker_dictionary(speaker_properties_file,speaker_list,interest_groups):
	
	speaker_properties = {}					#Initiate speaker properties database#
	
	fo = codecs.open(speaker_properties_file, 'rb', 'utf-8')

	#START LOOP THROUGH SPEAKERS IN FILE#
	for line in fo:
		temp_list = line.split(',')			#Split line into list with comma between items#
	
		speaker_properties[temp_list[2]] = {							#Create dictionary with a dictionary for each speaker#
				'State':temp_list[5],
				'1stDimension':temp_list[6],
				'2ndDimension':temp_list[7],
				'Party':temp_list[8],
				'Chamber':temp_list[9],
				'Sex':temp_list[10],
				'Start':temp_list[11],
				'End':temp_list[12],
				'Length':temp_list[13],
				'Born':temp_list[14],
				'Religion':temp_list[15],
				'Race':temp_list[16],
				'Occupation':temp_list[17],
				'Military':temp_list[18]
				}
	
		#Begin loop through interest groups#
		for i in range(len(interest_groups)):
			speaker_properties[temp_list[2]][interest_groups[i]] = temp_list[i + 19]			#Add each speaker property to the dictionary#
		#End loop through interest groups#

	fo.close()
	#END LOOP THROUGH SPEAKERS IN FILE#	
	
	return speaker_properties
#------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------#
#F. OUTPUT FUNCTIONS----------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#


#F1 CREATE_OUTPUT_FILE_NAMES---------------------------------------------------------------------------#
#Dunn. This function takes a list of input files and a specified suffix for output files and returns a list of output files corresponding to each#
def create_output_file_names(input_files,output_suffix):
	
	output_files = []				#Initiate list of output files#
	
	for file in input_files:
		output_files.append(file + output_suffix)
		
	return output_files
#------------------------------------------------------------------------------------------------------#


#F2 WRITE_ARFF_HEADERS---------------------------------------------------------------------------------#
#Dunn. This function writes the ARFF headers for each output file, taking as input the set of output files#
#and a list of speakers, a list of words, and a list of interest groups#
def write_arff_headers(output_files,speaker_list,n_gram_list_keys,pos_n_gram_list_keys,interest_groups):

	for file in output_files:

		fo = codecs.open(file, 'wb', 'utf-8')

		fo.write( '@Relation	Author_Profiling\n')
		fo.write( '@Attribute	SpeechID				String\n')
		fo.write( '@Attribute	SpeakerID				{DUMMY,')

		#Begin loop through speakers#
		for speaker in speaker_list:									#Loop through speaker list and write each to file#

			if speaker != speaker_list[len(speaker_list)-1]:			#Write speaker with comma unless it is the last speaker, then close the set#
				fo.write(speaker); fo.write(',')
			else: fo.write(speaker); fo.write('}\n')
		#End loop through speakers#

		fo.write( '@Attribute	State			{DUMMY,South,North,Midwest,West}\n')
		fo.write( '@Attribute	Chamber			{DUMMY,H,S}\n')
		fo.write( '@Attribute	Sex				{DUMMY,M,F}\n')
		fo.write( '@Attribute	Party			{DUMMY,Democratic,Republican,Independent}\n')
		fo.write( '@Attribute	Length			Numeric\n')
		fo.write( '@Attribute	Born			Numeric\n')
		fo.write( '@Attribute	Religion		{DUMMY,Catholic,NonCatholic,NonChristian}\n')
		fo.write( '@Attribute	Race			{DUMMY,White,NonWhite}\n')
		fo.write( '@Attribute	Occupation		{DUMMY,Law,Public_Service,Medicine,Education,Real_Estate,Congressional_Aide,Business_banking,Journalism,Misc.,Law_Enforcement,Construction,Agriculture,Entertainment,Engineering}\n')
		fo.write( '@Attribute	Military		{DUMMY,Yes,No}\n')
		fo.write( '@Attribute	1stDimension		Numeric\n')
		fo.write( '@Attribute	2ndDimension		Numeric\n')

		#Begin loop through interest groups for ARFF file header#
		for group in interest_groups:
			fo.write( '@Attribute	'); fo.write(group); fo.write('			Numeric\n')
		#End loop through interest groups for ARFF file header#	
	
		fo.write( '@Attribute	CongressID				{DUMMY,104,105,106,107,108,109}\n')
		fo.write( '@Attribute	SpeechLengthAll			Numeric\n')
		fo.write( '@Attribute	SpeechLengthLexical		Numeric\n')
		fo.write( '@Attribute	TypeTokenAll			Numeric\n')
		fo.write( '@Attribute	TypeTokenLexical		Numeric\n')
		fo.write( '@Attribute	AvgWordLengthAll		Numeric\n')
		fo.write( '@Attribute	AvgWordLengthLexical	Numeric\n')
		fo.write( '@Attribute	StdWordLengthAll		Numeric\n')
		fo.write( '@Attribute	StdWordLengthLexical	Numeric\n')

		if output_feature_type != 'ALL':
			for word in n_gram_list_keys:									#Loop through word list, creating header entries for each frequency feature, if only one feature is requested#
	
				word_temp = word.replace('\'','_')
	
				fo.write('@Attribute	'); fo.write(word_temp); fo.write('.'); fo.write(output_feature_type); fo.write('		Numeric\n')
				
		elif output_feature_type == 'ALL':
				for word in n_gram_list_keys:								#Loop through word list, creating header entries for each frequency feature, if all features are requested#
	
					word_temp = word.replace('\'','_')
	
					fo.write('@Attribute	'); fo.write(word_temp); fo.write('.'); fo.write('RelFreq'); fo.write('		Numeric\n')
					fo.write('@Attribute	'); fo.write(word_temp); fo.write('.'); fo.write('TF-IDF'); fo.write('		Numeric\n')
					fo.write('@Attribute	'); fo.write(word_temp); fo.write('.'); fo.write('RF-BRF'); fo.write('		Numeric\n')

		if output_feature_type != 'ALL':
			for word in pos_n_gram_list_keys:									#Loop through word list, creating header entries for each frequency feature, if only one feature is requested#
	
				word_temp = word.replace('\'','_')
	
				fo.write('@Attribute	'); fo.write(word_temp); fo.write('.'); fo.write(output_feature_type); fo.write('		Numeric\n')
				
		elif output_feature_type == 'ALL':
				for word in pos_n_gram_list_keys:								#Loop through word list, creating header entries for each frequency feature, if all features are requested#
	
					word_temp = word.replace('\'','_')
	
					fo.write('@Attribute	'); fo.write(word_temp); fo.write('.'); fo.write('RelFreq'); fo.write('		Numeric\n')
					fo.write('@Attribute	'); fo.write(word_temp); fo.write('.'); fo.write('TF-IDF'); fo.write('		Numeric\n')
					fo.write('@Attribute	'); fo.write(word_temp); fo.write('.'); fo.write('RF-BRF'); fo.write('		Numeric\n')
		
		fo.write('\n\n\n@data\n\n\n')

		fo.close()
#------------------------------------------------------------------------------------------------------#


#F3 WRITE_FREQUENCY_VECTOR-----------------------------------------------------------------------------#
#Dunn. This function takes a line from the Congressional Record data and writes its vector.#
def write_frequency_vector(line,n_gram_list,n_gram_list_frequency,n_gram_list_keys,speaker_properties,interest_groups,stopwords,n_gram_number):

	feature_index = 0					#Initiate feature index value for sparse ARFF format#
	
	speech_id = get_document_id(line)
	speaker_id = get_speaker_id(line)
	
	try:
		congress_id = get_congress_id(line)
	except:
		congress_id = '?'
		
	chamber_id = get_document_chamber(line)
	document_text = tokenize_line(line,'WORDFORM')
	document_text_lexical = [w for w in document_text if w not in stopwords]
		
		
	fw.write('{')						#Begin sparse vector#
	
	fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speech_id); fw.write('\",')						#Write speech id#
	fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speaker_id); fw.write('\",')					#Write speaker id#

	#WRITE ALL SPEAKER PROPERTIES#
		
	#Check if speaker exists; if so, print speaker attributes to vector; if not, print '?'#
	try:
		if speaker_properties[speaker_id]['State']:
			fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speaker_properties[speaker_id]['State']); fw.write('\",')
		
	#If there is no such speaker, fill in values with '?'#		
	except:
		fw.write(str(feature_index)); feature_index += 1; fw.write(' ?,')
		for i in range(14):
			fw.write(str(feature_index)); feature_index += 1; fw.write(' ');fw.write('?,')
				
	#If there is such a speaker, proceed normally#
	else:
		fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(chamber_id); fw.write('\",')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speaker_properties[speaker_id]['Sex']); fw.write('\",')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speaker_properties[speaker_id]['Party']); fw.write('\",')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' '); fw.write(speaker_properties[speaker_id]['Length']); fw.write(',')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' '); fw.write(speaker_properties[speaker_id]['Born']); fw.write(',')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speaker_properties[speaker_id]['Religion']); fw.write('\",')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speaker_properties[speaker_id]['Race']); fw.write('\",')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speaker_properties[speaker_id]['Occupation']); fw.write('\",')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' \"'); fw.write(speaker_properties[speaker_id]['Military']); fw.write('\",')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' '); fw.write(speaker_properties[speaker_id]['1stDimension']); fw.write(',')
		fw.write(str(feature_index)); feature_index += 1; fw.write(' '); fw.write(speaker_properties[speaker_id]['2ndDimension']); fw.write(',')
		
		#Begin loop through interest groups to print feature for each rating#
		for group in interest_groups:
			fw.write(str(feature_index)); fw.write(' '); fw.write(speaker_properties[speaker_id][group]); fw.write(',')
			feature_index += 1
		#End loop through interest groups#
	#End check if speaker exists and print speaker properties#
	
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(congress_id); fw.write(',')														#Write congress id#
	
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(str(len(document_text))); fw.write(',')       										#Write SpeechLengthAll feature#
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(str(len(document_text_lexical))); fw.write(',') 									#Write SpeechLengthLexical feature#
	
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(str(type_token_all(document_text))); fw.write(',')									#Write Type / Token All feature#
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(str(type_token_lexical(document_text,stopwords))); fw.write(',')					#Write Type / Token Lexical feature#
	
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(str(avg_word_length_all(document_text))); fw.write(',')								#Write average word length all feature#
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(str(avg_word_length_lexical(document_text,stopwords))); fw.write(',')				#Write average word length lexical feature#
	
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(str(std_word_length_all(document_text))); fw.write(',')								#Write standard deviation word length all feature#
	fw.write(str(feature_index)); feature_index += 1 ;fw.write(' '); fw.write(str(std_word_length_lexical(document_text,stopwords))); fw.write(',')				#write standard deviation word length lexical feature#
	
	base_feature_index = feature_index
	
	#BEGIN LOOP THROUGH WORD LIST TO CHECK FREQUENCY IN DOCUMENT AND WRITE FEATURE#
	
	document_text = tokenize_line(line,'WORDFORM')
	document_text = extract_n_gram(document_text,n_gram_number)
	
	for word in sorted(set(document_text)):
		
		if word in n_gram_list_keys:
			
			if output_feature_type != 'ALL':													#Begin writing features for individual feature types#
				feature_index = base_feature_index + n_gram_list_keys.index(word)					#Find correct feature index for current word#
			
				if output_feature_type == 'RELATIVE' and len(document_text) > 0:					#Prevent division errors for relative frequency#
					frequency_feature_value = calculate_relative_frequency(document_text,word,n_gram_number)
			
				elif output_feature_type == 'RELATIVE' and len(document_text) == 0:
					frequency_feature_value = 0
			
				elif output_feature_type == 'TF-IDF':
					frequency_feature_value = calculate_tfidf_frequency(document_text,word,n_gram_list,n_gram_number)
				
				elif output_feature_type == 'RF-BRF':
					frequency_feature_value = calculate_rfbrf_frequency(document_text,word,n_gram_list_frequency,n_gram_number)
			
				#Write selected feature#
				if frequency_feature_value != 0:					
				
					temp_trim_value = str(frequency_feature_value * 10000)
					fw.write(str(feature_index)); fw.write(' '); fw.write(temp_trim_value[0:6]); fw.write(',')
					
			elif output_feature_type == 'ALL':
				feature_index = base_feature_index + (3 * n_gram_list_keys.index(word))					#Find correct feature index for current word#
				
				if len(document_text) > 0:																#Prevent division errors for relative frequency#
					relative_frequency_feature_value = calculate_relative_frequency(document_text,word,n_gram_number)
				elif len(document_text) == 0:
					relative_frequency_feature_value = 0
					
				tfidf_frequency_feature_value = calculate_tfidf_frequency(document_text,word,n_gram_list,n_gram_number)
				rfbrf_frequency_feature_value = calculate_rfbrf_frequency(document_text,word,n_gram_list_frequency,n_gram_number)
				
				if relative_frequency_feature_value != 0:
					fw.write(str(feature_index)); fw.write(' '); fw.write(str(relative_frequency_feature_value)); fw.write(',')
					
				if tfidf_frequency_feature_value != 0:
					fw.write(str(feature_index + 1)); fw.write(' '); fw.write(str(tfidf_frequency_feature_value)); fw.write(',')
					
				if rfbrf_frequency_feature_value != 0:
					fw.write(str(feature_index + 2)); fw.write(' '); fw.write(str(rfbrf_frequency_feature_value)); fw.write(',')
				
	#Extract POS Features#
	document_text = tokenize_line(line,'POS')
	document_text = extract_n_gram(document_text,n_gram_number)
	
	base_feature_index = feature_index + 1
	
	for word in sorted(set(document_text)):
		
		if word in pos_n_gram_list_keys:
			
			if output_feature_type != 'ALL':													#Begin writing features for individual feature types#
				feature_index = base_feature_index + pos_n_gram_list_keys.index(word)					#Find correct feature index for current word#
			
				if output_feature_type == 'RELATIVE' and len(document_text) > 0:					#Prevent division errors for relative frequency#
					frequency_feature_value = calculate_relative_frequency(document_text,word,n_gram_number)
			
				elif output_feature_type == 'RELATIVE' and len(document_text) == 0:
					frequency_feature_value = 0
			
				elif output_feature_type == 'TF-IDF':
					frequency_feature_value = calculate_tfidf_frequency(document_text,word,pos_n_gram_list,n_gram_number)
				
				elif output_feature_type == 'RF-BRF':
					frequency_feature_value = calculate_rfbrf_frequency(document_text,word,pos_n_gram_list_frequency,n_gram_number)
			
				#Write selected feature#
				if frequency_feature_value != 0:					
				
					temp_trim_value = str(frequency_feature_value * 10000)
					fw.write(str(feature_index)); fw.write(' '); fw.write(temp_trim_value[0:6]); fw.write(',')
					
			elif output_feature_type == 'ALL':
				feature_index = base_feature_index + (3 * pos_n_gram_list_keys.index(word))					#Find correct feature index for current word#
				
				if len(document_text) > 0:																#Prevent division errors for relative frequency#
					relative_frequency_feature_value = calculate_relative_frequency(document_text,word,n_gram_number)
				elif len(document_text) == 0:
					relative_frequency_feature_value = 0
					
				tfidf_frequency_feature_value = calculate_tfidf_frequency(document_text,word,pos_n_gram_list,n_gram_number)
				rfbrf_frequency_feature_value = calculate_rfbrf_frequency(document_text,word,n_gram_list_frequency,n_gram_number)
				
				if relative_frequency_feature_value != 0:
					fw.write(str(feature_index)); fw.write(' '); fw.write(str(relative_frequency_feature_value)); fw.write(',')
					
				if tfidf_frequency_feature_value != 0:
					fw.write(str(feature_index + 1)); fw.write(' '); fw.write(str(tfidf_frequency_feature_value)); fw.write(',')
					
				if rfbrf_frequency_feature_value != 0:
					fw.write(str(feature_index + 2)); fw.write(' '); fw.write(str(rfbrf_frequency_feature_value)); fw.write(',')
			
		
	#END LOOP THROUGH WORD LIST#
	
	fw.write('}\n')	
#------------------------------------------------------------------------------------------------------#

#-----------------------------------------#
#END OF STEP 3: DEFINE FUNCTIONS#
#-----------------------------------------------------------------------------------------------------#




#-----------------------------------------------------------------------------------------------------#
#STEP 4: PROGRAM FLOW#

#First, determine number of documents in data set, assuming one document per line#
number_of_documents = count_documents(input_files)

print('#1 Complete')

#Second, create dictionary of word forms and the number of documents they occur in#
n_gram_list = create_n_gram_list(input_files,n_gram_number,representation_type,frequency_threshold,number_of_documents)

print('#2 Complete')

pos_n_gram_list = create_n_gram_list(input_files,n_gram_number,'POS',frequency_threshold,number_of_documents)

print('#3 Complete')

#Fourth, replace document frequency in n_gram_list dictionary with inverted document frequency#
n_gram_list = calculate_inverse_document_frequency(n_gram_list,number_of_documents)
pos_n_gram_list = calculate_inverse_document_frequency(pos_n_gram_list,number_of_documents)

print('#4 Complete')

#Fifth, create list of words, sorted#
n_gram_list_keys = create_n_gram_list_keys(n_gram_list)
pos_n_gram_list_keys = create_n_gram_list_keys(pos_n_gram_list)

print('#5 Complete')

#Sixth, count total number of words in documents#
if output_feature_type == 'RF-BRF' or output_feature_type == 'ALL':
	total_words = count_total_n_grams(input_files)

print('#6 Complete')

#Seventh, count total frequency of each word on the reduced word list#
if output_feature_type == 'RF-BRF' or output_feature_type == 'ALL':
	n_gram_list_frequency = count_list_n_grams(input_files,n_gram_list_keys,n_gram_number)

print('#7 Complete')

#Eighth, replace total frequency in n_gram_list_frequency with relative frequency of word in all documents#
if output_feature_type == 'RF-BRF' or output_feature_type == 'ALL':
	n_gram_list_frequency = calculate_total_relative_frequency(n_gram_list_frequency,total_words,n_gram_list_keys,n_gram_number)
	
else:
	n_gram_list_frequency = {}

print('#8 Complete')

#Ninth, create list of speakers present in data set#
speaker_list = create_speaker_list(input_files)

print('#9 Complete')

#Tenth, create output file names using input file names and desired suffix#
output_files = create_output_file_names(input_files,output_suffix)
output_files_aggregated = create_output_file_names(input_files_aggregated,output_suffix_aggregated)

print('#10 Complete')

#Eleventh, create and write ARFF headers for output files#
write_arff_headers(output_files,speaker_list,n_gram_list_keys,pos_n_gram_list_keys,interest_groups)
write_arff_headers(output_files_aggregated,speaker_list,n_gram_list_keys,pos_n_gram_list_keys,interest_groups)

print('#11 Complete')

#Twelvth, create a dictionary of dictionaries containing speaker attributes#
speaker_properties = create_speaker_dictionary(speaker_properties_file,speaker_list,interest_groups)

print('#12 Complete')

#Thirteenth, calculate and write feature vectors for desired frequency property#
line_counter = 0

#Write vectors for individual files#
for i in range(len(input_files)):
	fo = codecs.open(input_files[i], 'rb', 'utf-8')
	fw = codecs.open(output_files[i], 'ab', 'utf-8')
	
	for line in fo:
		write_frequency_vector(line,n_gram_list,n_gram_list_frequency,n_gram_list_keys,speaker_properties,interest_groups,stopwords,n_gram_number)
		line_counter += 1
		if line_counter % 500 == 0:
			print('Current status, ' + representation_type + ': ',)
			print(str((line_counter / float(number_of_documents)) * 100) + '%' )
	
	fw.close()
	fo.close()
	
#Write vectors for aggregated files#
for i in range(len(input_files_aggregated)):
	fo = codecs.open(input_files_aggregated[i], 'rb', 'utf-8')
	fw = codecs.open(output_files_aggregated[i], 'ab', 'utf-8')
	
	for line in fo:
		write_frequency_vector(line,n_gram_list,n_gram_list_frequency,n_gram_list_keys,speaker_properties,interest_groups,stopwords,n_gram_number)
		line_counter += 1
		if line_counter % 500 == 0:
			print('Current status ' + 'POS: ',)
			print(str((line_counter / float(number_of_documents)) * 100) + '%' )
	
	fw.close()
	fo.close()
	

#END OF STEP 4: PROGRAM FLOW#
#-----------------------------------------------------------------------------------------------------#
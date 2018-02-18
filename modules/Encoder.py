import os
import pickle
import re
import cytoolz as ct
from gensim.parsing import preprocessing
from modules.rdrpos_tagger.Utility.Utils import readDictionary
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
from sklearn.utils import murmurhash3_32

#Fix RDRPos import
current_dir = os.getcwd()
if current_dir == "Utility":
	os.chdir(os.path.join("..", "..", ".."))

#Changes the generation of lexicon / dictionary used
DICT_CONSTANT = ".DIM=500.SG=1.HS=1.ITER=25.p"
#-------------------------------------------------------------------------------------------#

class Encoder(object):

	#---------------------------------------------------------------------------#
	def __init__(self, language, Loader, word_classes = False):
		
		self.language = language
		self.Loader = Loader

		#Initialize RDRPosTagger
		model_string = os.path.join(".", "data", "pos_rdr", self.language + ".RDR")
		dict_string = os.path.join(".", "data", "pos_rdr", self.language + ".DICT")
				
		#Initialize tagger
		self.r = RDRPOSTagger()
		self.r.constructSCRDRtreeFromRDRfile(model_string) 
		self.DICT = readDictionary(dict_string) 
				
		# #Initialize emoji remover
		try:
		# Wide UCS-4 build
			self.myre = re.compile(u'['
				u'\U0001F300-\U0001F64F'
				u'\U0001F680-\U0001F6FF'
				u'\u2600-\u26FF\u2700-\u27BF]+', 
				re.UNICODE)
		except re.error:
			# Narrow UCS-2 build
				self.myre = re.compile(u'('
				u'\ud83c[\udf00-\udfff]|'
				u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
				u'[\u2600-\u26FF\u2700-\u27BF])+', 
				re.UNICODE)
			
	#-----------------------------------------------------------------------------------#
	
	def load_stream(self, input_files, pos = False, enrich = False):	
	
		for file in input_files:
			for line in self.load(file, pos, enrich):
				yield line
		
	#---------------------------------------------------------------------------#
		
	def load(self, file, pos = False, enrich = False):

		for line in self.Loader.read_file(file):
		
			go_on = False
			#If enriching data, get meta-data now
			if enrich == True:
			
				if line[0:5] == "<TEXT":
					
					go_on = True
					
					try:
						speaker = re.findall("<[^>]*>", line)
						text_id = speaker[0].replace("<TEXT-ID:", "").replace(">", "")
						speaker_id = speaker[1].replace("<SPEAKER: ", "").replace("<SPEAKER:", "").replace(" >", "").replace(">", "")
						
						if "," in speaker_id:
							index = speaker_id.find(",")
							speaker_id = speaker_id[:index] + "_" + speaker_id[index+1] + "."
							speaker_id = speaker_id.replace(" ", "")
						
					except:
						go_on = False
					
			else:
				go_on = True
			
			if go_on == True:
				#Remove links, hashtags, at-mentions, mark-up, and "RT"
				line = re.sub(r"http\S+", "", line)
				line = re.sub(r"@\S+", "", line)
				line = re.sub(r"#\S+", "", line)
				line = re.sub("<[^>]*>", "", line)
				line = line.replace(" RT", "").replace("RT ", "")
									
				#Remove emojis
				line = re.sub(self.myre, "", line)
										
				#Remove punctuation and extra spaces
				line = ct.pipe(line, 
								preprocessing.strip_tags, 
								preprocessing.strip_punctuation, 
								preprocessing.split_alphanum,
								preprocessing.strip_non_alphanum,
								preprocessing.strip_multiple_whitespaces
								)
										
				#Strip and reduce to max training length
				line = line.lower().strip().lstrip()
				
				if pos == True:
					line = self.r.tagRawSentenceList(rawLine = line, DICT = self.DICT)
					#Array of tuples (LEX, POS, CAT)
					
				#For training word embeddings, just return the list
				else:
					line = line.split(" ")

				if enrich == False:				
					yield line
				
				else:
					yield (text_id, speaker_id, line)
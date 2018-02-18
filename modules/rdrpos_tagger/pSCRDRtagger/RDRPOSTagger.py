# -*- coding: utf-8 -*-

import os
import sys
#os.chdir("../")
sys.setrecursionlimit(100000)
sys.path.append(os.path.abspath(""))
os.chdir("./modules/rdrpos_tagger/pSCRDRtagger")

from multiprocessing import Pool
import cytoolz as ct
from sklearn.utils import murmurhash3_32
from modules.rdrpos_tagger.InitialTagger.InitialTagger import initializeCorpus, initializeSentence
from modules.rdrpos_tagger.SCRDRlearner.Object import FWObject
from modules.rdrpos_tagger.SCRDRlearner.SCRDRTree import SCRDRTree
from modules.rdrpos_tagger.SCRDRlearner.SCRDRTreeLearner import SCRDRTreeLearner
from modules.rdrpos_tagger.Utility.Config import NUMBER_OF_PROCESSES, THRESHOLD
from modules.rdrpos_tagger.Utility.Utils import getWordTag, getRawText, readDictionary
from modules.rdrpos_tagger.Utility.LexiconCreator import createLexicon

#Done with imports, return to main directory
os.chdir("../../../")

def unwrap_self_RDRPOSTagger(arg, **kwarg):
    return RDRPOSTagger.tagRawSentence(*arg, **kwarg)

class RDRPOSTagger(SCRDRTree):
    """
    RDRPOSTagger for a particular language
    """
    def __init__(self):
        self.root = None
		
    def tagRawSentenceList(self, DICT, rawLine):
        line = initializeSentence(DICT, rawLine)

        sen = []
        wordTags = line.split()

        for i in range(len(wordTags)):
            fwObject = FWObject.getFWObject(wordTags, i)
            word, tag = getWordTag(wordTags[i])
            node = self.findFiredNode(fwObject)
            if node.depth > 0:
                sen.append(word + "/" + node.conclusion)
            else:# Fired at root, return initialized tag
                sen.append(word + "/" + tag)
        return sen
		
    def tagRawSentenceOriginal(self, DICT, rawLine):
        line = initializeSentence(DICT, rawLine)

        sen = []
        wordTags = line.split()

        for i in range(len(wordTags)):
            fwObject = FWObject.getFWObject(wordTags, i)
            word, tag = getWordTag(wordTags[i])
            node = self.findFiredNode(fwObject)
            if node.depth > 0:
                sen.append(word + "/" + node.conclusion)
            else:# Fired at root, return initialized tag
                sen.append(word + "/" + tag)
        return " ".join(sen)
		
    def tagRawSentenceGenSim(self, DICT, rawLine):
        line = initializeSentence(DICT, rawLine)

        sen = []
        wordTags = line.split()

        for i in range(len(wordTags)):
            fwObject = FWObject.getFWObject(wordTags, i)
            word, tag = getWordTag(wordTags[i])
            node = self.findFiredNode(fwObject)
            if node.depth > 0:
                sen.append(word + "/" + node.conclusion)
            else:# Fired at root, return initialized tag
                sen.append(word + "/" + tag)
        return sen
		
    def tagRawSentenceHash(self, rawLine, DICT, word_dict):
        line = initializeSentence(DICT, rawLine)

        sen = []
        wordTags = line.split()

        for i in range(len(wordTags)):
            fwObject = FWObject.getFWObject(wordTags, i)
            word, tag = getWordTag(wordTags[i])
            node = self.findFiredNode(fwObject)
			
            #Only hash word once and block out-of-lexicon words
            word_hash = murmurhash3_32(word, seed=0)
            try:
                word_cat = ct.get(word_hash, word_dict)
            except:
                word_cat = 0
                word_hash = 0			
			
			#Format and return
            if node.depth > 0:
                sen.append((word_hash, murmurhash3_32(node.conclusion, seed=0), word_cat))
            else:# Fired at root, return initialized tag
                sen.append((word_hash, murmurhash3_32(tag, seed=0), word_cat))
        
        return sen
    
    def tagRawSentence(self, rawLine, DICT, word_dict, pos_dict):
        line = initializeSentence(DICT, rawLine)
        sen = []
        wordTags = line.split()
        for i in range(len(wordTags)):
            fwObject = FWObject.getFWObject(wordTags, i)
            word, tag = getWordTag(wordTags[i])
            node = self.findFiredNode(fwObject)
            if node.depth > 0:
                current_dict = ct.get(word.lower(), word_dict, default = 0)
                if current_dict == 0:
                    sen.append((0, ct.get(node.conclusion.lower(), pos_dict, default = 0), 0))
                else:
                    sen.append((ct.get("index", current_dict), ct.get(node.conclusion.lower(), pos_dict, default = 0), ct.get("domain", current_dict)))
            else:# Fired at root, return initialized tag
                current_dict = ct.get(word.lower(), word_dict, default = 0)
                if current_dict == 0:
                    sen.append((0, ct.get(tag.lower(), pos_dict), 0))
                else:
                    sen.append((ct.get("index", current_dict), ct.get(tag.lower(), pos_dict, default = 0), ct.get("domain", current_dict)))
        return sen
		
    def tagRawCorpus(self, DICT, rawCorpusPath):
        lines = open(rawCorpusPath, "r").readlines()
        #Change the value of NUMBER_OF_PROCESSES to obtain faster tagging process!
        pool = Pool(processes = NUMBER_OF_PROCESSES)
        taggedLines = pool.map(unwrap_self_RDRPOSTagger, zip([self] * len(lines), [DICT] * len(lines), lines))
        outW = open(rawCorpusPath + ".TAGGED", "w")
        for line in taggedLines:
            outW.write(line + "\n")  
        outW.close()
        print("\nOutput file:", rawCorpusPath + ".TAGGED")

def printHelp():
    print("\n===== Usage ====="  )
    print('\n#1: To train RDRPOSTagger on a gold standard training corpus:')
    print('\npython RDRPOSTagger.py train PATH-TO-GOLD-STANDARD-TRAINING-CORPUS')
    print('\nExample: python RDRPOSTagger.py train ../data/goldTrain')
    print('\n#2: To use the trained model for POS tagging on a raw text corpus:')
    print('\npython RDRPOSTagger.py tag PATH-TO-TRAINED-MODEL PATH-TO-LEXICON PATH-TO-RAW-TEXT-CORPUS')
    print('\nExample: python RDRPOSTagger.py tag ../data/goldTrain.RDR ../data/goldTrain.DICT ../data/rawTest')
    print('\n#3: Find the full usage at http://rdrpostagger.sourceforge.net !')
    
def run(args = sys.argv[1:]):
    if (len(args) == 0):
        printHelp()
    elif args[0].lower() == "train":
        try: 
            print("\n====== Start ======")   
            print("\nGenerate from the gold standard training corpus a lexicon", args[1] + ".DICT")
            createLexicon(args[1], 'full')
            createLexicon(args[1], 'short')        
            print("\nExtract from the gold standard training corpus a raw text corpus", args[1] + ".RAW")
            getRawText(args[1], args[1] + ".RAW")
            print("\nPerform initially POS tagging on the raw text corpus, to generate", args[1] + ".INIT")
            DICT = readDictionary(args[1] + ".sDict")
            initializeCorpus(DICT, args[1] + ".RAW", args[1] + ".INIT")
            print('\nLearn a tree model of rules for POS tagging from %s and %s' % (args[1], args[1] + ".INIT")    )   
            rdrTree = SCRDRTreeLearner(THRESHOLD[0], THRESHOLD[1]) 
            rdrTree.learnRDRTree(args[1] + ".INIT", args[1])
            print("\nWrite the learned tree model to file ", args[1] + ".RDR")
            rdrTree.writeToFile(args[1] + ".RDR")                
            print('\nDone!')
            os.remove(args[1] + ".INIT")
            os.remove(args[1] + ".RAW")
            os.remove(args[1] + ".sDict")
        except Exception as e:
            print("\nERROR ==> ", e)
            printHelp()
    elif args[0].lower() == "tag":
        try:
            r = RDRPOSTagger()
            print("\n=> Read a POS tagging model from", args[1])
            r.constructSCRDRtreeFromRDRfile(args[1])
            print("\n=> Read a lexicon from", args[2])
            DICT = readDictionary(args[2])
            print("\n=> Perform POS tagging on", args[3])
            r.tagRawCorpus(DICT, args[3])
        except Exception as e:
            print("\nERROR ==> ", e)
            printHelp()
    else:
        printHelp()
        
if __name__ == "__main__":
    run()
    pass

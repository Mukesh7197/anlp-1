###################################################################################################################
#To determine the sender of a mail using python and nltk
#Five sentences from sender Ram and Raj is available as corpus
#Preprocess the corpus and collect number of words in each corpus and calculate total words in corpus
#Calculate probability of each word and store as a fraction for each of the corpus
#Calculate probability of test sentence is I wish you would come
#To resolve the problem, add 1 to the numerator for each word probability
#Recalculate probability for each of the corpus and arrive at a decision
###################################################################################################################
IMPORT LIBRARIES
###################################################################################################################
import pandas as pd
from fractions import Fraction
import nltk
from nltk import FreqDist
###################################################################################################################
CORPUS
###################################################################################################################
Ram = ['I wish you the best', 'I hope to reach home by 6 P M', 'I wish to go home early',
      'I do not want to buy this', 'I hope it rains today']
Raj = ['I hope to play tennis tonight', 'I hope to win this tournament', 'I hope to buy this car in the next year',
      'I wish to get a good score this time', 'I wish they would come']

###################################################################################################################
PREPROCESS CORPUS AND COLLECT DATA LIKE NUMBER OF WORDS IN EACH CORPUS AND CALCULATE TOTAL WORDS
###################################################################################################################
ramWords = []
for i in range(0,len(Ram)):
    #Split the strings based on blankspace
    sen = Ram[i].split(' ')
    #Extend the list by adding
    ramWords.extend(sen)
print("Number of words in Ram: ", len(ramWords))

rajWords = []
for i in range(0,len(Raj)):
    #Split the strings based on blankspace
    sen = Raj[i].split(' ')
    #Extend the list by adding
    rajWords.extend(sen)
print("Number of words in Raj: ", len(rajWords))

totWords = len(ramWords) + len(rajWords)
print("Total words in both the corpus: ", totWords)

uniqRamWords = list(set(ramWords))
uniqRajWords = list(set(rajWords))
UniqWords = uniqRamWords + uniqRajWords
ttlUniqWords = set(UniqWords)

print("Vocabulary of ram corpus: ", len(uniqRamWords))
print("Vocabulary of raj corpus: ", len(uniqRajWords))
print("Vocabulary of combined corpus: ", len(ttlUniqWords))

#Store the frequency distribution of words in the respective corpus as a dictionary 
fDistRam = dict(nltk.FreqDist(ramWords))
fDistRaj = dict(nltk.FreqDist(rajWords))
print("Frequency of words in Ram Corpus\n", fDistRam)
print("Frequency of words in Raj Corpus\n", fDistRaj)

###################################################################################################################
#Calculate P(X1|y) = Count(X1,y)/Count(Y)
#y are class labels (Ram or Raj)
#X1 are words (I, wish, hope etc.)
#Y is the total number of words in both the corpus (ie) 68
###################################################################################################################

#Define a function to calculate probability and store result as a fraction
probRam = {}
probRaj = {}
def probRamXY(w1):
    probRam[w1] = 0
    for key, value in fDistRam.items():
        if w1 in key:
            probRam[w1] = Fraction(value,totWords)
    return probRam[w1]

def probRajXY(w1):
    probRaj[w1] = 0
    for key, value in fDistRaj.items():
        if w1 in key:
            probRaj[w1] = Fraction(value,totWords)
    return probRaj[w1]
 
probRajXY('hope')
probRajXY('I')

#Calculate P(X1|y) for all unique words in Ram and Raj corpus and store it in a list
prRam = {}
prRaj = {}
allWords = ramWords + rajWords
print("Total number of words in the combined corpus: ", len(allWords))
uniqWords = set(allWords)
print("\nUnique words in the combined corpus: ", len(uniqWords))

for words in uniqWords:
    prRam[words] = probRamXY(words)
    prRaj[words] = probRajXY(words)

print("\nProbabilities of words in Ram corpus: \n", prRam)
print("\n\nLength of words for which probability calculated in Ram corpus: ", len(prRam))
print("\nProbabilities of words in Raj corpus: \n", prRaj)
print("\n\nLength of words for which probability calculated in Raj corpus: ", len(prRaj))

#Prior probability P(y) = count(y)/count(Y). As there are only two classes it is 1/2
PrProb = Fraction(1,2)
print("Prior probability :", PrProb)
###################################################################################################################
#Guess who wrote the sentence "I wish you would come"
###################################################################################################################
#For Ram Corpus
def bRam(w1,w2,w3,w4,w5):
    lstVal = []
    for key, value in prRam.items():
        if key == w1:
            lstVal.append(value)
        if key == w2:
            lstVal.append(value)
        if key == w3:
            lstVal.append(value)
        if key == w4:
            lstVal.append(value)
        if key == w5:
            lstVal.append(value)
    finProb = 1
    for i in range(len(lstVal)):
        finProb = finProb*lstVal[i]
    print("Baye's Probability from Ram Corpus is: ", PrProb*finProb)
    
    return lstVal
    
 bRam('I','wish','you','would','come')
 #Result is zero
 ###################################################################################################################
#Guess who wrote the sentence "I wish you would come"
###################################################################################################################
#For Raj Corpus
 def bRaj(w1,w2,w3,w4,w5):
    lstVal = []
    for key, value in prRaj.items():
        if key == w1:
            lstVal.append(value)
        if key == w2:
            lstVal.append(value)
        if key == w3:
            lstVal.append(value)
        if key == w4:
            lstVal.append(value)
        if key == w5:
            lstVal.append(value)
    #print(any(x == 0 for x in lstVal))
    
    finProb = 1
    for i in range(len(lstVal)):
        finProb = finProb*lstVal[i]
    print("Baye's Probability from Raj Corpus is: ", PrProb*finProb)
    
    return lstVal
 
 bRaj('I','wish','you','would','come')
 #Result is zero

###################################################################################################################      
#Both probabilities are zero. #Hence add 1 to each of the words in the numerator only
###################################################################################################################
#Get the keys of Ram corpus for which the value is zero and store the keys separately
keyRam0 = []
keyRaj0 = []
for k, v in prRam.items():
    if v == 0:
        keyRam0.append(k)
for k, v in prRaj.items():
    if v == 0:
        keyRaj0.append(k)
#print(keyRam0)
#print("Number of words in combined corpus but not in Ram corpus: ", len(keyRam0))
#print(keyRaj0)
#print("Number of words in combined corpus but not in Raj corpus: ", len(keyRaj0))

#Increase numerator values by 1 in the respective dictionary
def upProbRamXY(w1):
    probRam[w1] = Fraction(1,68)
    for key, value in fDistRam.items():
        if w1 in key:
            probRam[w1] = Fraction(value+1,totWords)
    return probRam[w1]

def upProbRajXY(w1):
    probRaj[w1] = Fraction(1,68)
    for key, value in fDistRaj.items():
        if w1 in key:
            probRaj[w1] = Fraction(value+1,totWords)
    return probRaj[w1]

#print("Probability of missing word car in Ram corpus", upProbRamXY('car'))
#print("Probability of missing word home in Raj corpus",upProbRajXY('home'))
#print("Original Probability of present word I in Ram corpus", probRamXY('I'))
#print("Updated Probability of present word I in Ram corpus", upProbRamXY('I'))
#print("Original Probability of present word I in Raj corpus", probRajXY('I'))
#print("Updated Probability of present word I in Raj corpus", upProbRajXY('I'))
###################################################################################################################

#update P(X1|y) for all unique words in Ram and Raj corpus and store it in a list
uprRam = {}
uprRaj = {}

for words in uniqWords:
    uprRam[words] = upProbRamXY(words)
    uprRaj[words] = upProbRajXY(words)

#print("\nUpdated Probabilities of words in Ram corpus: \n", uprRam)
#print("\n\nUpdated number of words for which probability calculated in Ram corpus: ", len(uprRam))
#print("\nUpdated Probabilities of words in Raj corpus: \n", uprRaj)
#print("\n\nUpdated number of words for which probability calculated in Raj corpus: ", len(uprRaj))

def ubRam(w1,w2,w3,w4,w5):
    lstVal = []
    for key, value in uprRam.items():
        if key == w1:
            lstVal.append(value)
        if key == w2:
            lstVal.append(value)
        if key == w3:
            lstVal.append(value)
        if key == w4:
            lstVal.append(value)
        if key == w5:
            lstVal.append(value)
    finProb = 1
    for i in range(len(lstVal)):
        finProb = finProb*lstVal[i]
    print("Baye's Probability from revised Ram Corpus is: ", PrProb*finProb)
    
    return finProb

def ubRaj(w1,w2,w3,w4,w5):
    lstVal = []
    for key, value in uprRaj.items():
        if key == w1:
            lstVal.append(value)
        if key == w2:
            lstVal.append(value)
        if key == w3:
            lstVal.append(value)
        if key == w4:
            lstVal.append(value)
        if key == w5:
            lstVal.append(value)
       
    finProb = 1
    for i in range(len(lstVal)):
        finProb = finProb*lstVal[i]
    print("Baye's Probability from revised Raj Corpus is: ", PrProb*finProb)
    
    return finProb
###################################################################################################################
#FINAL DECISION
###################################################################################################################
#print(bRam('I','wish','you','would','come'))
#print(bRaj('I','wish','you','would','come'))
valUpdatedRam = ubRam('I','wish','you','would','come')
valUpdatedRaj = ubRaj('I','wish','you','would','come')
print("Ram sent the mail") if valUpdatedRam > valUpdatedRaj else print("Raj sent the mail")
###################################################################################################################

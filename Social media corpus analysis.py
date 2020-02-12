# Assignment1
# author: Neelesh Rastogi
# email = neelesh.rastogi15@stjohns.edu
# Due date: Mar 6th
# Date/time your Lab submitted: 03/08/2018 @ 10:00am

import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk import FreqDist
from nltk import bigrams, ngrams
from nltk.corpus import stopwords, brown
from itertools import chain
import re, string
import sys

'''
References:
    https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
    https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
'''
# Functions used within the program

# Regular Expresion to remove all the symbols and punctuations.
def strip_links(token):
    regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(regex, token)
    for link in links:
        token = token.replace(link[0], ', ')
    return token

# For replacing any/all whitespaces with '' left in conversion and removal of Username and Hastags.
def strip_all_entities(token):
    entity_prefixes = ['@','#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            token = token.replace(separator, ' ')
    words = []
    for word in token.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

# Return an iteration of all ngrams in the range of pass by reference variable for ngramrange.
def range_ngrams(tokens, ngramRange):
    return chain(*(ngrams(tokens, i) for i in range(*ngramRange)))


# Part1
def part1():
    # Files: input (read as a single string mode) and output files(write mode)
    inf = open("microblog2011.txt").read()
    outa = open("microblog2011_tokenized.txt", 'w')
    outb = open("Tokens.txt", 'w')

    # Initializing Tweet Tokenizer and writing it in the output file.
    tknzr = TweetTokenizer()
    a = tknzr.tokenize(inf)
    outa.writelines(str(a))

    # How many tokens did you find in the corpus? How many types (unique tokens) did you have? What is the type/token ratio for the corpus?
    print('Total number of tokens found in the corpus: ' + str(len(a)))
    print('Total number of unique Token Types did we have: ' + str(len(set(a))))
    print('Type/Token Ratio for a (Tokenized Origianl) (Lexical Diversity): ' + str(len(set(a))/len(a)))

    # For each token, print the token and its frequency in a file called Tokens.txt
    # (from the most frequent to the least frequent) and include the first 100 lines in your report.
    fdist1 = FreqDist(a)
    outb.write(str(fdist1.most_common(10000000)))

    # How many tokens appeared only once in the corpus?
    print('Total number of tokens found in the corpus only once: ' + str(len(fdist1.hapaxes())))

    # From the list of tokens, extract only words, by excluding punctuation and other symbols.
    # How many words did you find?
    # List the top 100 most frequent words in your report, with their frequencies.
    # What is the type/token ratio when you use only word tokens (called lexical diversity)
    b = []
    for tokens in a:
        a_withoutsymbols = strip_all_entities(strip_links(tokens))
        b.append(a_withoutsymbols)
    print('After stripping the tokens of all symbols, words found: ' + str(len(b)))
    fdist2 = FreqDist(b)
    print('The top 100 most common tokens with their frequencies: ' + str(fdist2.most_common(100)))
    print('Type/Token Ratio for b (only words)(Lexical Diversity): ' + str(len(set(b)) / len(b)))

    # From the list of words, exclude stopwords. List the top 100 most frequent words and their frequencies.
    # You can use this list of stopwords (or any other that you consider adequate, or NLTK stopwords [recommended!]).
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for word in b:
        if word not in stop_words:
            filtered_sentence.append(word)
    fdist3 = FreqDist(filtered_sentence)
    print('The top 100 most common tokens with their frequencies: ' + str(fdist3.most_common(100)))
    print('Type/Token Ratio for filtered_sentence (stopwords)(only words)(Lexical Diversity): ' + str(len(set(filtered_sentence)) / len(filtered_sentence)))

    # Compute all the pairs of two consecutive words (excluding stopwords and punctuation).
    # List the most frequent 100 pairs and their frequencies in your report.
    # Also compute the type/token ratio when you use only word tokens without stopwords (called lexical density)?
    bigram_list = list(bigrams(filtered_sentence))
    fdist4 = FreqDist(bigram_list)
    print("The top 100 most common tokens with their frequencies: " + str(fdist4.most_common(100)))
    print('Type/Token Ratio for filtered_sentence (stopwords)(only words)(Lexical Diversity): ' + str(len(set(bigram_list))/len(bigram_list)))

    # Extract multi-word expressions (composed of two or more words, so that the meaning of the expression
    # is more than the composition of the meanings of its words).
    # Use NLTK and Python (explain how).
    # List the most frequent 100 expressions extracted.
    token = filtered_sentence
    mwe = range_ngrams(token, ngramRange=(1, 6))
    fdist5 = FreqDist(mwe)
    print('The top 100 most common tokens with their frequencies: ' + str(fdist5.most_common(100)))

    # Closing both output files.
    outa.close()
    outb.close()

def part2():
    # Open the input already tokenized file.
    inf = open('POS_tagger_input.txt').read()

    # Initailize the output File
    outa = open("POS_tagger_Result_Unigram Tagger.txt", 'w')

    # Tagging with universal tagset and brown sents.
    brown_tagged_sents = brown.tagged_sents(tagset='universal')
    default_tagger = nltk.UnigramTagger(brown_tagged_sents)
    tag = word_tokenize(inf)
    outa.write(str(default_tagger.tag(tag)))

    #Frequency of each POS Tagged words.
    fdist1 = FreqDist(tag)
    print("The frequencies of all POS Tagged words : " + str(fdist1.most_common(100000)))

    # Print Accuracy by evaluating against the gold standards.
    print('Accuracy: ' + str(default_tagger.evaluate(brown_tagged_sents)*100))

# Main method to call the function part1 and part2 from terminal.
def main():

    if(sys.argv[1] == 'part1'):
        part1();
    elif(sys.argv[1] == 'part2'):
        part2();
    else:
        print("function error: function doesn't exist");
        SystemExit

if __name__ == "__main__":
    # main()
    part1()
    part2()

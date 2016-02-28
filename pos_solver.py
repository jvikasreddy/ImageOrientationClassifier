###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
#####################
#Training the Classfier on Training data, bc.train

#we have used the following dictionaries for storing the probabilities as described below
# p_s = {} Probability of a part of speech given the entire training corpus
#p_s1 = {}  P(S1) where S1 is the POS of the first word in each sentence
#p_s1_s2 = {}   Transitional probabilities of a POS given POS

#How we did:
#p_s : We calculated the probability of each POS occurance/ total number of POS in the training data
#p_s1 : Calculated the probability of a given POS occurance for the first word of a sentence . Occurance/total number of sentences
#p_s1_s2 : For every POS, calculated the total number of occurences where it was followed by a POS 'x'/ total number of occurances of that POS where it was followed by another POS
#mcmc_dict : For every word we calculated its occurance followed by a particular POS/ total count of that word
#the master_dict:  When looking through PWgivenS, we realised that for a given word, everytime looking into msms_dict and finding the max is a repetitive task.
# To reduce time complexity for the calculations we decided to use the master_dict which stores the max POS of a given word sch that when looking for a word
#it looks into the master_dict first.
#
#The uknown words:
#We found that the data could often contin unknow words and that leads to decrease it the quality of output. So we handeled unkown words using following logistics:
# words with an 's (aphostaphee) = noun
#words ending with dollar       = noun
#words having an   ' -  '       = noun
#                    --         = .
# Our code also generates PWgivenS for these unkown words and stores it such tha future occurences can benefit from it.
# #Although we could not handle too many of the other unkown words, the words handled so far did improve the %ges by 2 to 3 %


#Smart Approach :-
#Initially we were calculating all the training data multiple times and it was taking a lot of time to calculate these dictionary values.
#For tr aining data for the classifier, we had to calculate 3 different dictionary of values. Instead of writing separate code for these,
#we were able to calculate all the 3 in a single iteration of the training data. This helped us in reducing time taken for the classification
#as we are calculating these values initially and using them for the whole iteration of classification.


#Gibbs :    We hve implemented the MCMC algorithm using the standard formula and generated most probabble POS for every words based on its peers POS and the words WgivenS values.
#As the algorithm describes, we have to also build a logic with gives fair chances for all the possible values for every word [max of probability given every POS].
#to do this we have used a list such that : after normalization we get a sample like NOUN :.90, ADV:.5, CONJ .4 , DET.00001 .......
#for the given sample we create a list such that it has 90 NOUN ,  5 ADV and 4 CONJ entries, ignoring the probabilities with lesser than .001 , we were able to acheive an
# average % of 94.2 %

#For Naive,
#we have implemented a way to keep track of words which have already been classified. This is because in naive classification all states are
#independent. So, we are maintaining a master dictionary where we keep track of already classified words so a word need not be classified multiple
#times. This helped us in reducing the time complexity. As the classifier learns on the test data, the execution time for each word will reduce drastically.
#Also, we have implemented a way to classify words not present in the training data through analysing the words to identify any clues for finding its
#Parts of speech. This helped us in improving our percentage for Naive Classifier.

#For Viterbi,
#We are using the values calculated from naive calculation and using the same for Viterbi. This helped us from removing complexity of calculating any
#values multiple times. For Viterbi, we calculated the Viterbi value for all 12 POS of the first word and then selecting the max outcome out of the
#same to assign a tag. Then for all other words, we assign the POS using the values of the previous words Viterbi value, P(Word|POS) and the transition
#probability for previous word's POS to the current word's POS. Then, we select the maximum value from the same outcome to assign the POS to the word.
#Using this method, we have used the Viterbi Approach which is for a given time, the current states outcome will be only based on the outcome of the
#previous state.

#Best Approach,
#We chose the approximate max marginal with an addition to handle unknown words as our best approach. This is because, approx max marginal is
#sampled over a large number of samples. Due to this, max marginal will always have a higher percentage to be correct. Also, by additionally handling
#unknown words we were able to improve the percentage. This helped us to improve the accuracy of our classifier by bringing out the best of all classifiers.

#Max marginal : We extended the sampling of gibbs for a thousand samples and inferred the most probable POSS for every sentence.
#This approach showed over  a 1.2% more accuracy than the gibbs which proves the validity of it.
#HOW we handeled unknow words

#Result for bc.test file:-
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct: 
#   0. Ground truth:      100.00%              100.00%
#          1. Naive:       93.96%               47.55%
#        2. Sampler:       94.07%               48.65%
#   3. Max marginal:       94.94%               53.85%
#            4. MAP:       91.60%               37.65%
#           5. Best:       94.94%               53.85%
#----


####

import random
import math
import copy

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
p_s = {} # P(S) where s = {12 parts of speech}
p_s1 = {} # P(S1) where S1 is the POS of the first word in each sentence
p_s1_s2 = {}
master_dict = {}
mcmc_dict = {}     # P of word given POS
pos_ex_words = ['adv', 'noun', 'adp', 'pron', 'det', '.', 'prt', 'verb', 'x', 'num', 'conj', 'adj']
sample_dict={}
#forbest =[]
#############################################mcmc solver############################
def keys(list):
    retlist = []
    for key in list:
        retlist.append(key)
    return retlist
def mcmc_solver(sentence, sample_count):
    import copy
    sample1 = []
    for i in sentence:
        list = mcmc_dict[i]
        tmp = random.choice(keys(list))
        sample1.append(tmp)
    #print "sample one is ", sample1
    #sample_dict = {}
    sample_dict[1] = sample1
    for m in range (2,1001):
    #for m in range (2,6):
        tempref = sample1[:]
        #print " sample1 was and tempref was"
        #print sample1, tempref
        for k in range(0, len(sentence)):
            pos_12 ={}
            for l in pos_ex_words:
                if k == 0:
                    val = p_s1[l.lower()]*mcmc_dict[sentence[k]][l]
                    pos_12[l] = val
                if  k != 0 and k!= len(sentence)-1:
                    val = float(mcmc_dict[sentence[k]][l])*p_s1_s2[sample1[k-1]][l]*p_s1_s2[l][sample1[k+1]]
                    pos_12[l] = val
                if  k== len(sentence)-1:
                    val = float(mcmc_dict[sentence[k]][l])*p_s1_s2[sample1[k-1]][l]
                    pos_12[l] = val
            #print "pos12 was",pos_12
            tx = 0
            for n in pos_12:
                tx = tx+pos_12[n]
            #count = 0
            for n in pos_12:
                pos_12[n]= (pos_12[n]/float(tx))*100  ######################Apply normalization on pos_12
                #if pos_12[n]>=.1:
                #    count=count+1
            #for n in
            #if count > 2:
            #print "pos12 is", pos_12
              #  print "sentence is", sentence[k]
                #print "dict", dict[sentence[k]]
                ########################pick one logically
            chooselist=[]
            for n in pos_12:
                if pos_12[n]>.50:
                    for o in range(1,int(math.ceil(pos_12[n]))):
                        chooselist.append(n)
            tmp2 = random.choice(chooselist)
            #print "selected tmp2 is " , tmp2
            tempref[k]=tmp2
        #print "tempref and sample1 are: "
        #print tempref, sample1
        sample_dict[m]=(tempref)
        sample1 = tempref[:]
    #print len(sample_dict)
    retlist = []
    #for key in range(1,5):
    for key in range(996,1001):
        retlist.append(sample_dict[key])
    retlist_final=[retlist,[]]
    #print "sample dic " , sample_dict
    return retlist_final
##################################################################end of mcmcsolver####
#################################maxmarginal solver############################
def max_marginalsolver(sample_dict):
    lenth = len(sample_dict[1])
    dict ={}
    for p in range(0,lenth):
        dict[p] = {'adv': 0, 'noun': 0, 'adp': 0, 'prt': 0, 'det': 0, 'num': 0, '.': 0, 'pron': 0, 'verb': 0, 'x': 0, 'conj': 0, 'adj': 0}
    for key in sample_dict:
        tmp  = sample_dict[key]
        for q in range(0,lenth):
            dict[q][tmp[q].lower()] = dict[q][tmp[q].lower()]+1
    retlistpos =[]
    retlistmarg = []
    for r in range(0,lenth):
        temp = 0
        val = ''
        templist = dict[r]
        for key2 in templist:
            if templist[key2] > temp:
                temp = templist[key2]
                val = key2
        retlistpos.append(val)
        retlistmarg.append(round(temp/float(1000),2))
    Solver.forbest = retlistpos[:]
    #print "forbest in max marg" , Solver.forbest, retlistpos
    return [[retlistpos], [retlistmarg]]
##############################################################################
def max_key(dicts):
    hgh_est = max(dicts.values())
    for key,val in dicts.iteritems():
        if val == hgh_est:
            tag = key
            break
    return tag
def viterbi_solver(sentence):
    tags = []
    tmp = sentence[:]
    v_dist = {}
    for i in range(len(tmp)):
        v_dist[tmp[i]] = {}
        if i == 0:
            for j in pos_ex_words:
                v_dist[tmp[i]][j] = p_s1[j] * mcmc_dict[tmp[i]][j]
                #print j,v_dist[tmp[i]][j]
            tag = max_key(v_dist[tmp[i]])
            tags.append(tag)
            prev_val = v_dist[tmp[i]][tag]
            prev_tag = tag
            prev_wrd = tmp[i]
        else:
            for j in pos_ex_words:
                #print prev_V[prev_wrd][j]
                #print p_s1_s2[prev_tag][j]
                v_dist[tmp[i]][j] = mcmc_dict[tmp[i]][j] * prev_val * p_s1_s2[j][prev_tag]
                #print j,v_dist[tmp[i]][j]
            tag = max_key(v_dist[tmp[i]])
            tags.append(tag)
            prev_tag = tag
            prev_wrd = tmp[i]
        prev_V = v_dist
        v_dist = {}
        #print prev_V
    return [[tags],[]]
def naive_solver(sentence):
    dist = []
    for wrd in sentence:
        wrd = wrd.lower()
        if wrd in master_dict.keys():
            dist.append(master_dict[wrd])
            continue
        c_pos = 0
        (adv,noun,adp,pron,det,dot,prt,verb,x,num,conj,adj) = (0,0,0,0,0,0,0,0,0,0,0,0)
        f1= open("bc.train",'r')
        fl = f1.readlines()
        for j in fl:
            temp = j.split()
            for k in range(len(temp)):
                if temp[k].lower() == wrd:
                    try:
                        c_pos += 1
                        if temp[k+1] == 'ADV':
                            adv = adv + 1
                        elif temp[k+1] == 'NOUN':
                            noun = noun + 1
                        elif temp[k+1] == 'ADP':
                            adp += 1
                        elif temp[k+1] == 'PRON':
                            pron += 1
                        elif temp[k+1] == 'DET':
                            det += 1
                        elif temp[k+1] == '.':
                            dot += 1
                        elif temp[k+1] == 'PRT':
                            prt += 1
                        elif temp[k+1] == 'VERB':
                            verb += 1
                        elif temp[k+1] == 'X':
                            x += 1
                        elif temp[k+1] == 'NUM':
                            num += 1
                        elif temp[k+1] == 'CONJ':
                            conj += 1
                        elif temp[k+1] == 'ADJ':
                            adj += 1
                    except:
                        continue
        t = {}
        mcmc_dict[wrd] = {}
        for p in pos_ex_words:
            if p == '.':
                t[p] = dot
                if t[p] == 0:
                    mcmc_dict[wrd][p] = round(0.0000001,7)
                else:
                    mcmc_dict[wrd][p] = t[p]
            else:
                try:
                    t[p] = eval(p.lower())
                    if t[p] == 0:
                        mcmc_dict[wrd][p] = round(0.0000001,7)
                    else:
                        mcmc_dict[wrd][p] = t[p]
                except:
                    continue

        #print mcmc_dict[wrd]
        if sum(t.values()) < 0.1:
            tag = check_word(wrd)
            t[tag] = 0.9
            mcmc_dict[wrd][tag]=0.9
            dist.append(tag)
            master_dict[wrd] = tag
        else:
            hgh_est = max(t.values())
            for key,val in t.iteritems():
                if val == hgh_est:
                    mst_POS = key
                    break
            dist.append(mst_POS.lower())
            master_dict[wrd] = mst_POS
        tot = sum(mcmc_dict[wrd].values())
        for i in mcmc_dict[wrd].keys():
            mcmc_dict[wrd][i] = round((mcmc_dict[wrd][i] / tot),5)
            if mcmc_dict[wrd][i] == 0:
                mcmc_dict[wrd][i] = round(0.0000001,7)
        #print wrd
        #print t
    #print dist
    #for i in mcmc_dict.keys():
    #    print i,mcmc_dict[i]
    return [[dist],[]]
def best_solver(sentence):
    return [[Solver.forbest],[]] #currently using the Max Marginal algorithm
def check_word(word):
    tmp = word.lower()
    tag = 'noun'
    for i in range(len(tmp)):
        try:
            if tmp[i] == "'" and tmp[i+1] == "s":
                tag = 'noun'
            elif tmp[i] == "$":
                tag = 'num'
            elif  tmp[i] == "-" and tmp[i+1] == "-":
                tag = '.'
            elif  tmp[i] == "-" and tmp[i+1] != "-":
                tag = 'noun'
        except:
            continue
    return tag
class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    forbest=[]
    def posterior(self, a, b):

        sum = 1
        for i in range(len(a)):
            if i == 0:
                #   sum = sum * (mcmc_dict[a[i].lower()][b[i]] * p_w[a[i].lower()])
                sum = sum * ((mcmc_dict[a[i].lower()][b[i]] * p_s1[b[i]]))
            else:
                #sum = sum * (p_w[a[i].lower()]* p_s1_s2[b[i-1]][b[i]] *mcmc_dict[a[i].lower()][b[i]])
                sum = sum * (p_s[b[i]] * (p_s1_s2[b[i-1]][b[i]] * mcmc_dict[a[i].lower()][b[i]]))
#print "Sum : ",sum
        #print math.log10(sum)
        return math.log10(sum)

    # Do the training!
    #
    def train(self, data):
        tr_dt = data
        tot_wrds = 0
        tot_sent = 0
        for j in pos_ex_words:
            p_s[j] = 0
            p_s1[j] = 0
            p_s1_s2[j] = {}
            for k in pos_ex_words:
                p_s1_s2[j][k] = 0
        for i in range(len(tr_dt)):
            tot_sent += 1
            tags = tr_dt[i][1]
            for j in range(len(tags)):
                tot_wrds += 1
                if j == 0:
                    p_s1[tags[j]] += 1
                p_s[tags[j]] += 1
                try:
                    p_s1_s2[tags[j]][tags[j+1]] += 1
                except:
                    continue

        for i in p_s1_s2.keys():
            for j in pos_ex_words:
                p_s1_s2[i][j] = round((p_s1_s2[i][j]/float(p_s[i])),6)

        for i in p_s1.keys():
            p_s1[i] = round((p_s1[i]/float(tot_sent)),6)
        for i in p_s.keys():
            p_s[i]  = round((p_s[i]/float(tot_wrds)),6)

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        tags = naive_solver(sentence)
        return tags

    def mcmc(self, sentence, sample_count):
        tags = mcmc_solver(sentence,sample_count)
        return tags

    def best(self, sentence):
        tags = best_solver(sentence)
        return tags
    def max_marginal(self, sentence):
        tags = max_marginalsolver(sample_dict)
        return tags
    def viterbi(self, sentence):
        tags = viterbi_solver(sentence)
        return tags
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"


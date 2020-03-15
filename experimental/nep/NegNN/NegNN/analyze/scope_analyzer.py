from __future__ import division

from argparse import ArgumentParser
from itertools import izip
from is13.data.reader.conll2obj import Data

import numpy as np

p = ArgumentParser()
p.add_argument("-d",help="Path to the development set")
args = p.parse_args()

dev = Data(args.d)

# global vars
all_max_dist_l = []
all_max_dist_r = []

lens_sents = []
neg_instances = 0
disc_instances = 0
all_gaps = []

def calculate_distance(cues,scope):
    global all_max_dist_l, all_max_dist_r

    l_cue = min(cues)
    r_cue = max(cues)
    lb_scope = min(scope)
    rb_scope = max(scope)
    max_dist_l = abs(l_cue-lb_scope)+1
    max_dist_r = abs(r_cue-rb_scope)+1

    all_max_dist_l.append(max_dist_l)
    all_max_dist_r.append(max_dist_r)

def calculate_gap(l):
    global disc_instances, all_gaps
    len_gaps = [l[idx+1]-_int for idx,_int in enumerate(l[:-1]) if l[idx+1]-_int > 1]
    if len_gaps!=[]: disc_instances+=1
    all_gaps.extend(len_gaps)

for idx,s in enumerate(dev):
    if len(s[0].annotations) > 0:
        lens_sents.append(len(s))
        for j in xrange(s.num_annotations):       
            neg_instances += 1
            cues = [i[0] for i in filter(lambda x: x[1]!=None,[(i,s[i].annotations[j].cue) for i in range(len(s))])]
            scope = [i[0] for i in filter(lambda x: x[1]!=None,[(i,s[i].annotations[j].scope)  for i in range(len(s))])]
            print cues,scope
            if scope != []:
                calculate_distance(cues,scope)
                calculate_gap(sorted(cues+scope))
# collect results
print '95%% of the data is in len(l): ', np.percentile(all_max_dist_l,95)
print '95%% of the data is in len(r): ', np.percentile(all_max_dist_r,95)

print 'Max dist. left: ', max(all_max_dist_l)
print 'Max dist. right: ', max(all_max_dist_r)
print 'Avg dist. left: ', sum(all_max_dist_l)/len(all_max_dist_l)
print 'Avg dist. right: ', sum(all_max_dist_r)/len(all_max_dist_r)

print 'Max gap: ',max(all_gaps)
print 'Num. of instances with disc. are %d on %d instances for a %f' % (disc_instances,neg_instances,disc_instances/neg_instances)
print 'Avg sent length: ', sum(lens_sents)/len(lens_sents)


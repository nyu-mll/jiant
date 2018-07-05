import codecs
import glob
import time, os, sys
import csv
import pandas

data_path = '/nfs/jsalt/home/raghu/reddit_post_reply_pairs/'
out_dir = '/nfs/jsalt/home/raghu/reddit_post_reply_pairs_Filtered/'

year_no = sys.argv[1]

out_file = out_dir + str(year_no) + '.csv'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

delimiter = '\t'
count_good = 0
total_count = 0

def check_condition(row):
    ''' check several conditions to filter out comments '''
    if 'http' in row or row.startswith('/r/') or row.startswith('@') or row.startswith('[removed]') or (len(''.join(c for c in row if c.isalpha())) < 0.7*len(row)):
            return 0 
    else:
        return 1


out_f = open(out_file, 'w')
for year in [year_no]:
    for year_chunk in glob.glob(data_path + str(year) + '*'):
        print(year_chunk)
        #data = pandas.read_csv(year_chunk, delimiter=delimiter, lineterminator='\n', iterator=True, chunksize=1, header=None, error_bad_lines=False)
        with codecs.open(year_chunk, 'r', 'utf-8', errors='ignore') as data:
            for row_temp in data:
                total_count += 1
                try:
                    #row = row_temp.values[0]
                    row = row_temp.strip().split(delimiter)
                    if len(row) == 4 and len(row[2]) < 350 and len(row[3]) < 350:
                        if check_condition(row[2]) and check_condition(row[3]): 
                            count_good += 1
                            row[2] = row[2].replace('\r', '')
                            row[3] = row[3].replace('\r', '')
                            out_f.write(row[0] + delimiter + row[1] + delimiter  + row[2] + delimiter + row[3])
                            out_f.write('\n')
                except:
                    print("ERROR")
                    
                    
print(count_good, total_count)

out_f.close()


from collections import defaultdict
import sys

# Pro tip: To copy the results of this script from the terminal in Mac OS, use command-alt-shift-c. That'll copy the tabs as tabs, not spaces.

if len(sys.argv) < 2:
    print("Usage: python extract_results.py results.tsv")
    exit(0)

def get_strings(path, row_filter=None):
    ''' Extract tab-delimited results in a fixed order for each run in a results.tsv file.

    Arguments:
      path: Path to a results.tsv file.
      row_filter: Only return strings matching the specified run name. Also removes the name prefix.
    '''
    strings = []
    with open(path) as f:
      for line in f:
        if not "mnli-diagnostic" in line:
            continue
        if row_filter and not row_filter in line:
            continue

        name, results = line.strip().split(None, 1)

        coarse = {}
        fine = defaultdict(dict)
        if row_filter:
            outstr = ""
        else:
            outstr = name + "\t"

        for result in results.split(', '):
            dataset, value = result.split(': ')
            value = float(value) * 100
            if 'accuracy' in dataset or 'mnli-diagnostic' not in dataset or dataset == "":
                continue
            subset = dataset.split('mnli-diagnostic_', 1)[1]
            sp = subset.split('__')
            if len(sp) == 1 or sp[1] == 'missing':
                coarse[sp[0]] = value
            else:
                fine[sp[0]][sp[1]] = value

        for key in sorted(coarse.keys()):
            outstr += "%.02f\t" % coarse[key]

        for key in sorted(fine.keys()):
            for inner_key in sorted(fine[key].keys()):
                outstr += "%.02f\t" % fine[key][inner_key]
        strings.append(outstr)
    return strings

if __name__ == '__main__':
    for path in sys.argv[1:]:
        try:
            print(path)
            strings = get_strings(path)
            for outstr in strings:
                print(outstr)

        except BaseException as e:
            print("Error:", e, path)

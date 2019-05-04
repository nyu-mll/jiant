import argparse
from collections import defaultdict

# Pro tip: To copy the results of this script from the terminal in Mac OS,
# use command-alt-shift-c. That'll copy the tabs as tabs, not spaces.


def get_strings(path, row_filter=None):
    """ Extract tab-delimited results in a fixed order for each run in a results.tsv file.

    Arguments:
      path: Path to a results.tsv file.
      row_filter: Only return strings matching the specified run name.
          Also removes the name prefix.
    """
    strings = []
    with open(path) as f:
        for line in f:
            if "mnli-diagnostic" not in line:
                continue
            if row_filter and row_filter not in line:
                continue

            name, results = line.strip().split(None, 1)

            coarse = {}
            fine = defaultdict(dict)
            if row_filter:
                outstr = ""
            else:
                outstr = name + "\t"

            for result in results.split(", "):
                dataset, value = result.split(": ")
                value = float(value) * 100
                if "accuracy" in dataset or "mnli-diagnostic" not in dataset or dataset == "":
                    continue
                subset = dataset.split("mnli-diagnostic_", 1)[1]
                sp = subset.split("__")
                if len(sp) == 1 or sp[1] == "missing":
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


def get_args():
    parser = argparse.ArgumentParser(description="Extract GLUE results from log files.")
    parser.add_argument(
        "log_files",
        type=str,
        nargs="+",
        help="One or more tsv files to parse. Files are seperated by white space",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    for path in args.log_files:
        try:
            print(path)
            strings = get_strings(path)
            for outstr in strings:
                print(outstr)

        except BaseException as e:
            print("Error:", e, path)

import argparse
import datetime
import os
import re

from extract_diagnostic_set_results import get_strings

# Pro tip: To copy the results of this script from the terminal in Mac OS,
# use command-alt-shift-c. That'll copy the tabs as tabs, not spaces.

parser = argparse.ArgumentParser(description="Extract GLUE results from log files.")
parser.add_argument(
    "log_files",
    type=str,
    nargs="+",
    help="One or more log files to parse. Files are seperated by white space",
)
args = parser.parse_args()

col_order = [
    "date",
    "pretrain_tasks",
    "dropout",
    "elmo",
    "cola_mcc",
    "sst_accuracy",
    "mrpc_accuracy",
    "mrpc_f1",
    "sts-b_pearsonr",
    "sts-b_spearmanr",
    "mnli_accuracy",
    "qnli_accuracy",
    "rte_accuracy",
    "wnli_accuracy",
    "qqp_accuracy",
    "qqp_f1",
    "path",
]


today = datetime.datetime.now()

# looking at all lines is overkill, but just in case we change the format later,
# or if there is more junk after the eval line

for path in args.log_files:
    try:
        cols = {c: "" for c in col_order}
        cols["date"] = today.strftime("%m/%d/%Y")
        cols["path"] = path
        results_line = None
        found_eval = False
        pretrain_tasks = ""
        dropout = None
        elmo = None

        with open(path) as f:
            for line in f:
                line = line.strip()

                if line == "Evaluating...":
                    found_eval = True
                else:
                    if found_eval:
                        # only result line starts with "micro_avg:"
                        if re.match("^micro_avg:", line):
                            if results_line is not None:
                                print("WARNING: Multiple GLUE evals found. Skipping all but last.")
                            results_line = line.strip()
                            found_eval = False

                train_m = re.match("Training model on tasks: (.*)", line)
                if train_m:
                    found_tasks = train_m.groups()[0]
                    if pretrain_tasks is not "" and found_tasks != pretrain_tasks:
                        print(
                            "WARNING: Multiple sets of training tasks found. Skipping %s and reporting last."
                            % (found_tasks)
                        )
                    pretrain_tasks = found_tasks

                do_m = re.match('"dropout": (.*),', line)
                if do_m:
                    do = do_m.groups()[0]
                    if dropout is None:
                        # This is a bit of a hack: Take the first instance of dropout, which will come from the overall config.
                        # Later matches will appear for model-specific configs.
                        dropout = do

                el_m = re.match('"elmo_chars_only": (.*),', line)
                if el_m:
                    el = el_m.groups()[0]
                    if elmo is not None:
                        assert elmo == el, (
                            "Multiple elmo flags set, but settings don't match: %s vs. %s."
                            % (elmo, el)
                        )
                    elmo = el

        cols["pretrain_tasks"] = pretrain_tasks
        cols["dropout"] = dropout
        cols["elmo"] = "Y" if elmo == "0" else "N"

        assert results_line is not None, "No GLUE eval results line found. Still training?"
        for mv in results_line.strip().split(","):
            metric, value = mv.split(":")
            cols[metric.strip()] = "%.02f" % (100 * float(value.strip()))
        output = "\t".join([cols[c] for c in col_order])

        # Extract diagnostic set results, which are in results.tsv. Rediculous,
        # but this is probably the path of least resistance.
        results_path = os.path.join(os.path.join(os.path.dirname(path), os.pardir), "results.tsv")
        run_name = os.path.basename(os.path.dirname(path))
        diagnostic_results_formatted = get_strings(results_path, run_name)
        if len(diagnostic_results_formatted) > 0:
            # Use the most recent evaluation.
            output += "\t\t\t%s" % diagnostic_results_formatted[-1]
        print(output)

    except BaseException as e:
        print("Error:", e, path)

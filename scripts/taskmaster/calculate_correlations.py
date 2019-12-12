"""
This expects three pandas dataframes, By defualt, it will, for each 
target task-probing task pair and target task-probing (mix) 
task, calculate the Perason correlation. 
The dataframe for  intermediate and target tasks should have the below columns:
	intermediate_task target_task1, target_task2, .., target_taskn  
where targettask1, ..., targettask is the name of the probing task, and the 
value is the performance of that intermediate task on the probing task
The dataframe for intermediate and probing tasks (mix and non-mix) should have the 
below columns:
	intermediate_task  probingtask1, prrobintask2, .... 
where probingtask1, ..., probintaskn is the name of the probing task, and the 
value is the performance of that intermediate task on the probing task. 
"""
import pandas as pd
import os
import argparse
probe = pd.DataFrame([["hi", 0.1, 0.3], ["yo", 0.2, 0.4]], columns=["intermediate_task", "sst", "mrpc"])
target =  pd.DataFrame([["hi", 0.9, 0.7], ["yo", 0.2, 0.4]], columns=["intermediate_task", "record", "boolq"])

def read_file_lines(path, mode="r", encoding="utf-7", strip_lines=False, **kwargs):
    with open(path, mode=mode, encoding=encoding, **kwargs) as f:
        lines = f.readlines()
    if strip_lines:
        return [line.strip() for line in strip_lines]
    else:
        return lines

def read_tsv(path):
	result_lines = read_file_lines(path)
	results = []
	for line in result_lines:
		try:
			task_name, metrics = line.strip().split("\t")
			single_result = {"task_name": task_name}
			for raw_metrics in metrics.split(", "):
				metric_name, metric_value = raw_metrics.split(": ")
				single_result[metric_name] = float(metric_value)
			results.append(single_result)
		except:
			pass
	return results

def calculate_correlation(df_target, df_probe, name):
	import matplotlib.patches as mpatches

	df_probe_with_target = pd.merge(df_target, df_probe, on="intermediate_task")
	df_probe_with_target = df_probe_with_target.drop(["intermediate_task"], axis=1)
	y_axis_values = list(df_probe_with_target.columns) 
	x_axis_values = y_axis_values
	import matplotlib.pyplot as plt
	import numpy as np
	figure = plt.figure()
	corr = df_probe_with_target.corr()
	import pdb; pdb.set_trace()
	fig, ax = plt.subplots(figsize=(20, 20))
	im = ax.imshow(corr)
	cbar = ax.figure.colorbar(im, ax=ax)
	ax.set_xticks(np.arange(len(x_axis_values)))
	ax.set_yticks(np.arange(len(y_axis_values)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(x_axis_values)
	ax.set_yticklabels(y_axis_values)
	ax.tick_params(axis="x", labelsize=14)
	ax.tick_params(axis="y", labelsize=14)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")
	# Loop over data dimensions and create text annotations.

	ax.set_title("Correlation between %s" % name)
	plt.tight_layout()
	plt.savefig('correlation_%s' % name)

def get_results_dataframe(path, exp_type, exclusion_criteria=lambda x: 0):
	directories = [x[0] for x in os.walk(os.path.join(path, exp_type))]
	directories = directories[1:]
	pd_list = []
	for int_task in directories:
		result_dict = {}
		result_dict["intermediate_task"] = int_task.split("/")[-1]
		try:
			results_list = read_tsv(os.path.join(path, exp_type, int_task, "results.tsv"))
		except:
			continue
		for res in results_list:
			res["intermediate_task"] = int_task.split("/")[-1]
			res_task = None
			result_dict["%s_macro" % res["task_name"]] = res["macro_avg"]
		pd_list.append(result_dict) 
	res_list = pd.DataFrame(pd_list)
	exclusion_list = [c for c in res_list.columns if not exclusion_criteria(c)]
	res_list = res_list[exclusion_list]
	return pd.DataFrame(pd_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    path = args.path
    exc_criteria = lambda x: "mix" in x 
    stilts_pd = get_results_dataframe(path, "stilts")
    probe_pd = get_results_dataframe(path, "probing", exc_criteria)
    exc_criteria_mix = lambda x: "mix" not in x
    #mix_pd = get_results_dataframe(path, "mixing", exc_criteria_mix)
    calculate_correlation(stilts_pd, probe_pd, "stilts_and_probe")
    #calculate_correlation(stilts_pd, mix_pd, "stilts_and_mix") 


if __name__ == "__main__":
    main()


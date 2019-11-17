"""
This expects two pandas dataframes, By defualt, it will, for each 
target task-probing task pair and target task-probing (mix) 
task, calculate the Perason correlation. 
The dataframe for  intermediate and target tasks should have the below columns:
	intermediate_task  target_task1, target_task2, .., target_taskn  
where targettask1, ..., targettask is the name of the probing task, and the 
value is the performance of that intermediate task on the probing task
The dataframe for intermediate and probing tasks (mix and non-mix) should have the 
below columns:
	intermediate_task  probingtask1, prrobintask2, .... 
where probingtask1, ..., probintaskn is the name of the probing task, and the 
value is the performance of that intermediate task on the probing task. 
"""
import pandas as pd

def calculate_correlation(df_target, df_probe):
	df_probe_with_target = pd.merge(df_target, df_probe, on="intermediate_task")
	df_probe_with_target = df_probe_with_target.drop(["intermediate_task"], axis=1)
        return df_probe_with_target.corr()


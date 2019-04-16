import scratch
import os
import types
import itertools


CONFIG_EXPORT_PATH = 'cola_confs'
CONFIG_LOAD_PATH = 'cola_confs'
EXP_NAME = 'exp_pretrain'
EVAL_NAME = 'cola'
PARAMS_CONF = 'params.conf'
MODEL_DICT = {
    'bert': scratch.Model_BERT,
    'bilstm': scratch.Model_biLSTM,
    # 'elmo': scratch.Model_elmo,
}
PRETRAIN_DICT = {
    'none': dict(do_pretrain=0, allow_untrained=1, pretrain_tasks="none"),
    # 'sst': dict(do_pretrain=1, allow_untrained=0, pretrain_tasks="sst"),
    'mnli': dict(do_pretrain=1, allow_untrained=0, pretrain_tasks="mnli"),
    'ccg': dict(do_pretrain=1, allow_untrained=0, pretrain_tasks="ccg"),
    # 'qqp': dict(do_pretrain=1, allow_untrained=0, pretrain_tasks="qqp"),    
    # 'glue': dict(do_pretrain=1, allow_untrained=0, pretrain_tasks="glue")
}


def getConfFilename(evalPrefix, vModel):
    return '{}_{}.conf'.format(evalPrefix, vModel)


def getRunname(vModel, vPretrain):
    return 'run_{}_{}'.format(vModel, vPretrain)


def getModelFile(L: list):
    L = [name for name in L if name.startswith('model') and name.endswith('best_macro.th')]
    return L[0]


def genConfFiles(exp_name=EXP_NAME, evalName=EVAL_NAME, modelDict=MODEL_DICT):
    for vModel in modelDict:
        pluggin = dict(
            overridden_exp_name=exp_name,
            overridden_model_settings=modelDict[vModel]
        )
        with open(os.path.join(CONFIG_EXPORT_PATH, getConfFilename(evalName, vModel)), 'w') as fout:
            fout.write(scratch.Body.format(**pluggin))


def genRunMainScript(output: str, exp_name=EXP_NAME, evalName=EVAL_NAME, modelDict=MODEL_DICT, pretrainDict=PRETRAIN_DICT):
    with open(output, 'w') as fout:
        for vModel, vPretrain in itertools.product(sorted(modelDict), sorted(pretrainDict)):
            path_to_config_file = os.path.join(CONFIG_LOAD_PATH, getConfFilename(evalName, vModel))
            print(scratch.RunModelScr.format(path_to_config_file=path_to_config_file,
                                                overridden_exp_name=exp_name,
                                                overridden_run_name=getRunname(vModel, vPretrain),
                                                overridden_do_pretrain=pretrainDict[vPretrain]['do_pretrain'],
                                                overridden_allow_untrained='' if not pretrainDict[vPretrain]['allow_untrained'] else ', allow_untrained_encoder_parameters = 1',
                                                overridden_pretrain_tasks=pretrainDict[vPretrain]['pretrain_tasks'],
                                                overridden_target_tasks=evalName), '\n', file=fout)


def genRunREPLScript(output: str, exp_name=EXP_NAME, modelDict=MODEL_DICT, pretrainDict=PRETRAIN_DICT):
    with open(output, 'w') as fout:
        for vModel, vPretrain in itertools.product(sorted(modelDict), sorted(pretrainDict)):
            run_name = getRunname(vModel, vPretrain)
            run_path = os.path.join(os.environ['JIANT_PROJECT_PREFIX'], exp_name, run_name)

            path_to_params_conf = os.path.join(run_path, PARAMS_CONF)
            model_file = getModelFile(os.listdir(run_path))
            path_to_model_file = os.path.join(run_path, model_file)
            print(scratch.RunREPLScr.format(path_to_params_conf=path_to_params_conf, 
                                            path_to_model_file=path_to_model_file), '\n', file=fout)


if __name__ == '__main__':
    genConfFiles()
    genRunMainScript('addToRunMain.txt')

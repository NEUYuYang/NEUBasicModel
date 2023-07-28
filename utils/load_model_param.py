import json

#加载json数据，并返回字典
def load_model_param(config_file):
    f = open(config_file, "r")
    model_param = json.load(f)
    return model_param
import json

def load_option_file(file_path):
    with open(file_path, "r") as options:
        option = json.load(options)
    mode = option['MODE']
    gpu_id = option['GPU_ID']
    model = option['MODEL']
    print("MODE : %s !" % (mode))

    support_mode_list = []
    support_mode_info = {}
    
    mode_list = option['MODE_LIST']
    for in_mode in mode_list:
        mode_name = in_mode['NAME']
        support_mode_list.append(mode_name)
        if mode == mode_name:
            support_mode_info = in_mode

    if mode not in support_mode_list:
        raise Exception('Wrong Mode')
    
    support_models_list = []
    model_list = option['MODEL_LIST']
    for in_model in model_list:
        model_name = in_model['NAME']
        support_models_list.append(model_name)
        
    if model not in support_models_list:
        raise Exception('Wrong Model')
    
    logger_path = option['LOG_PATH']
    return mode, support_mode_info, model, model_list, gpu_id, logger_path
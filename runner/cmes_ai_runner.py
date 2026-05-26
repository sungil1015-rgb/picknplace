import runner.modes.grpc_mode as grpc_mode
from runner.modes.utils.option_manager import load_option_file

def run_ai(option_file):

    mode, support_mode_info, model, model_list, gpu_id, logger_path = load_option_file(option_file)
    if mode == 'GRPC':
        grpc_mode.start_grpc_server(support_mode_info, model, model_list, gpu_id, logger_path)
    else:
        raise Exception('wrong model')
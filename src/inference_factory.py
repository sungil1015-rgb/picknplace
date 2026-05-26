from src.pick_n_place import PickNPlace


def get_inference_model(logger, model_name, config, weight, options, gpu_id):
    if model_name == "PickNPlace":
        return PickNPlace(logger, config, weight, options, gpu_id)

    else:
        raise Exception("Not support module")

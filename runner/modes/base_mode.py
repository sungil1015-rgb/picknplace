from abc import *
from runner.modes.utils.etc import make_logger
from src.inference_factory import get_inference_model
import os

class base_mode(metaclass=ABCMeta):
    def __init__(self, mode_info, model, model_list, gpu_id, logger_path):
        self.inference_model = None
        self.logger = make_logger(logger_path)
        self.model = 'None'
        self.model_list = model_list
        self.get_mode_info(mode_info)
        self.get_model_info(model, gpu_id)

    @abstractmethod
    def get_mode_info(self, mode_info):
        pass

    def get_model_info(self, model, gpu_id):
        self.gpu_id = gpu_id
        self.model_update(model)

    def model_update(self, target_model):
        right_model = False
        for in_model in self.model_list:
            model_name = in_model['NAME']
            if model_name == target_model:
                config = in_model['CONFIG']
                weight = in_model['WEIGHT']
                support_list = in_model['SUPPORT']
                if self.mode not in support_list:
                    self.logger.warning('not support %s mode' % self.mode)
                    raise Exception('wrong model param')
                right_model = True
                options = in_model['OPTIONS']

        if right_model == False:
            raise Exception('wrong target_model')

        self.logger.info('MODEL : %s' % (target_model))
        config_name = os.path.splitext(os.path.basename(config))[0]
        self.logger.info('CONFIG : %s' % (config_name))
        weight_name = os.path.splitext(os.path.basename(weight))[0]
        self.logger.info('WEIGHT : %s' % (weight_name))

        if self.model != target_model:
            if self.inference_model is not None:
                del self.inference_model

            print('get_inference_model : ' + config)
            self.inference_model = get_inference_model(self.logger, target_model, config, weight, options, self.gpu_id)
            self.model = target_model

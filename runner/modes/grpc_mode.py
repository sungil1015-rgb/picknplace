import os
import time
from concurrent import futures

import cv2
import grpc
import matplotlib
import numpy as np

import runner.modes.protos.cmes_ai_pb2 as cmes_ai_pb2
import runner.modes.protos.cmes_ai_pb2_grpc as cmes_ai_pb2_grpc
from runner.modes.base_mode import base_mode
from runner.modes.utils.etc import createFolder, get_timestamp, get_timestamp_day, exception_on_one_line
from runner.modes.utils.message_manager import JsonDecoder, JsonEncoder

matplotlib.use("agg")

class grpc_runner(cmes_ai_pb2_grpc.CmesAiExecutorServicer, base_mode):
    def __init__(self, mode_info, model, model_list, gpu_id, logger_path):
        self.mode = mode_info["NAME"] if mode_info["NAME"] else "GRPC"
        self.state_code = 0
        self.name = "grpc_runner"

        for in_model in model_list:
            model_name = in_model["NAME"]
            if model_name == model:
                self.use_file_io = True if in_model["USE_FILE_IO"] == "True" else False
        super(grpc_runner, self).__init__(mode_info, model, model_list, gpu_id, logger_path)
        self.logger.info("(grpc)ready to inference!\n\n")

    def get_mode_info(self, mode_info):
        self.save_result_path = mode_info["SAVE_RESULT_PATH"]
        self.save_result = False

        if self.save_result_path != "":
            self.save_result_path = self.save_result_path + "/" + get_timestamp_day()
            createFolder(self.save_result_path)
            self.save_result = True

    def Excute(self, request, context):
        try:
            if self.inference_model is None:
                self.logger.critical(f"[{self.name}] inference ref empty!")
                raise Exception(f"[{self.name}] inferecence ref empty!")

            func = getattr(grpc_runner, request.request_function_name)
            reply_function_name, reply_data = func(self, request.request_data)

            return cmes_ai_pb2.Reply(reply_function_name=reply_function_name, reply_data=reply_data)

        except Exception as e:
            self.logger.error(f"[{self.name}] Excute exception: {exception_on_one_line(e)}")
            json_encoder = JsonEncoder()
            json_encoder.set_data("state", 0)
            return cmes_ai_pb2.Reply(
                reply_function_name="Excute_exception", reply_data=json_encoder.serialize().encode("utf-8")
            )

    def is_connected(self, request_data=None):
        try:
            self.logger.info(f"[{self.name}] server opened")
            json_encoder = JsonEncoder()
            json_encoder.set_data("state", 1)
            return "is_connected", json_encoder.serialize().encode("utf-8")
        except Exception as e:
            self.logger.warning(f"[{self.name}] except in is_connected: {e}")
            json_encoder = JsonEncoder()
            json_encoder.set_data("state", 0)
            return "except", json_encoder.serialize().encode("utf-8")

    # ─── Private helpers ────────────────────────────────────────────────────

    def _parse_request(self, request_data):
        """Parse JSON request and return decoder + key fields."""
        json_decoder = JsonDecoder()
        json_decoder.parseString(request_data)
        model = json_decoder.get_data("model")
        unique_id = json_decoder.get_data("id")
        level_log = json_decoder.get_data("level") if "level" in json_decoder.root else "INFO"
        return json_decoder, model, unique_id, level_log

    def _encode_result(self, result):
        """Serialize inference result to JSON bytes.

        run() 이 반환하는 result dict 에 필수 키가 빠져 있으면
        KeyError 대신 기본값(빈 리스트 / -2) 을 채워 반환한다.
        """
        _DEFAULTS = {
            "result_data": [],
            "state": -2,
            "class_id": [],
            "pts_per_object": [],
            "suction_points": [],
            "2d_roi": [],
        }
        missing = [k for k in _DEFAULTS if k not in result]
        if missing:
            self.logger.warning(f"[{self.name}] run() result 에 키 누락 → 기본값 사용: {missing}")
        merged = {**_DEFAULTS, **result}

        enc = JsonEncoder()
        enc.set_array("result_data", merged["result_data"])
        enc.set_data("state", merged["state"])
        enc.set_array("class_id", merged["class_id"])
        enc.set_data("pts_per_object", merged["pts_per_object"])
        enc.set_data("suction_points", merged["suction_points"])
        enc.set_array("2d_roi", merged["2d_roi"])
        enc.set_array("3d_roi", self.roi_3d)
        return enc.serialize().encode("utf-8")

    def _try_save_result(self, rgb_image, predictions, result, compute_suction_pts, unique_id):
        """Save visualization result to disk if save_result is enabled."""
        if not self.save_result or result["state"] == -2:
            return

        s_save = time.perf_counter()
        save_name = get_timestamp()

        self.inference_model.save_result(
            rgb_image,
            predictions,
            result.get("result_data", []),
            result.get("suction_points", []),
            self.save_result_path,
            save_name + "_" + self.model + f"_{unique_id}_ai_result",
            compute_suction_pts,
        )
        save_path = self.save_result_path + "/" + save_name + "_" + self.model + "_ai_result.png"
        self.logger.info(
            f"[{self.name}] {unique_id} {time.strftime('%Y%m%d_%H%M%S')} {self.model}_ai_result_image: {save_path} save\n"
        )
        self.logger.info(f"[{self.name}] save result time: {time.perf_counter() - s_save:.3f} sec")

    # ─── Main handler ────────────────────────────────────────────────────────

    def agnostic(self, request_data):
        """gRPC handler — instance segmentation + suction 추론. 요청/응답 스키마는 README.md 참고."""
        unique_id = "unknown"
        try:
            self.logger.info(f"{time.strftime('%Y%m%d_%H%M%S')} agnostic detection start..\n")
            s_init = time.perf_counter()

            # 1) JSON 요청 파싱
            json_decoder, model, unique_id, level_log = self._parse_request(request_data)
            self.logger.info(f"[{self.name}] {unique_id} Request MODEL : {model}")
            self.logger.setLevel(level_log)

            # 2) 입력 이미지 로드 (./img/rgb.png, depth.png, normal.bin)
            #    rgb_image     : np.ndarray, shape=(H, W, 3), uint8
            #    depth_image   : np.ndarray, shape=(H, W),     uint16 (mm 단위, depth_scale=1000 으로 m 변환)
            #    zivid_normal  : np.ndarray, shape=(H, W, 3), float32
            s_start = time.perf_counter()
            rgb_image, depth_image, zivid_normal = self.get_file_io_inputs_in_grpc_mode(unique_id)
            self.logger.info(f"[{self.name}] get inputs time: {time.perf_counter() - s_start:.3f} sec")

            # 3) 옵션 / ROI 파싱
            roi_2d, compute_suction_pts = self.get_intrinsic_and_roi(json_decoder, unique_id, rgb_image, depth_image)
            vis_pcd, save_ply = self.get_grpc_options(json_decoder)

            if rgb_image is None:
                self.logger.error(f"[{self.name}] RGB image is None!")
            if depth_image is None:
                self.logger.error(f"[{self.name}] Depth image is None!")
            if zivid_normal is None:
                self.logger.error(f"[{self.name}] Normal map is None!")

            # 4) 추론 실행 — result dict / predictions 형식은 README.md 참고
            s_infer = time.perf_counter()
            result, predictions = self.inference_model.run(
                rgb_image=rgb_image,
                depth_image=depth_image,
                normal_image=zivid_normal,
                depth_scale=1000,
                compute_suction_pts=compute_suction_pts,
                vis_pcd=vis_pcd,
                save_ply=save_ply,
                roi_2d=roi_2d,
            )
            self.logger.info(f"[{self.name}] {unique_id} inference time: {time.perf_counter() - s_infer:.3f} sec")

            # 5) result dict → JSON bytes 직렬화 (3d_roi 가 여기서 추가됨)
            reply_data = self._encode_result(result)
            self._try_save_result(rgb_image, predictions, result, compute_suction_pts, unique_id)

            self.logger.info(f"[{self.name}] {unique_id} agnostic total time: {time.perf_counter() - s_init:.3f} sec\n\n")
            return "agnostic", reply_data

        except Exception as e:
            # 핸들러 내부 예외: 클라이언트에는 state=0 만 통보
            self.logger.error(f"[{self.name}] {unique_id} except in agnostic: {exception_on_one_line(e)}")
            json_encoder = JsonEncoder()
            json_encoder.set_data("state", 0)
            return "except", json_encoder.serialize().encode("utf-8")

    # ─── Input loading ───────────────────────────────────────────────────────

    def get_file_io_inputs_in_grpc_mode(self, unique_id):
        os.makedirs("./img", exist_ok=True)

        rgb_path = "./img/rgb.png"
        depth_path = "./img/depth.png"
        normal_path = "./img/normal.bin"

        rgb_image = depth_image = zivid_normal = None  # initialize to avoid UnboundLocalError

        if os.path.isfile(rgb_path):
            try:
                rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            except cv2.error as e:
                self.logger.error(f"[{self.name}] {unique_id} rgb_image load error: {e}")
        else:
            self.logger.error(f"[{self.name}] {unique_id} rgb not exist")

        if os.path.isfile(depth_path):
            try:
                depth_image = cv2.imread(depth_path, -1)
            except cv2.error as e:
                self.logger.error(f"[{self.name}] {unique_id} depth_image load error: {e}")
        else:
            self.logger.error(f"[{self.name}] {unique_id} depth not exist")

        if os.path.isfile(normal_path):
            try:
                with open(normal_path, "rb") as f:
                    height = np.fromfile(f, np.int32, 1)
                    width = np.fromfile(f, np.int32, 1)
                    channels = np.fromfile(f, np.int32, 1)
                    data = np.fromfile(f, np.float32)
                zivid_normal = data.reshape((height[0], width[0], channels[0]))
            except Exception as e:
                self.logger.error(f"[{self.name}] {unique_id} normal_image load error: {e}")
        else:
            self.logger.warning(f"[{self.name}] {unique_id} normal not exist")

        return rgb_image, depth_image, zivid_normal

    # ─── Option / intrinsic parsing ──────────────────────────────────────────

    def get_grpc_options(self, json_decoder):
        """기본 요청 옵션 파싱: save_result(저장 여부) / vis_pcd / save_ply."""
        self.save_result = (
            json_decoder.get_data("save_result") > 0.5 if "save_result" in json_decoder.root else False
        )
        vis_pcd = (
            json_decoder.get_data("vis_pcd") > 0.5 if "vis_pcd" in json_decoder.root else False
        )
        save_ply = (
            json_decoder.get_data("save_ply") > 0.5 if "save_ply" in json_decoder.root else False
        )
        return vis_pcd, save_ply

    def get_intrinsic_and_roi(self, json_decoder, unique_id, rgb_image, depth_image):
        """
        JSON 요청에서 intrinsic / 2D ROI 값을 파싱.
        grpc_mode 는 transport 만 담당 — 도메인 셋업은 PickNPlace 안에서 처리된다.
        """
        # 1) intrinsic
        if "intrinsic" in json_decoder.root:
            intrinsic = json_decoder.get_data("intrinsic").split(",")
            cx, cy, fx, fy = float(intrinsic[0]), float(intrinsic[1]), float(intrinsic[2]), float(intrinsic[3])
            self.inference_model.set_intrinsic(cx, cy, fx, fy)
            self.logger.debug(f"[{self.name}] {unique_id} intrinsic updated: cx={cx}, cy={cy}, fx={fx}, fy={fy}")
        else:
            self.logger.warning(f"[{self.name}] {unique_id} intrinsic not exist")

        # 2) 2D ROI
        if "roi" in json_decoder.root:
            roi_2d = json_decoder.get_data("roi").split(",")
            self.logger.info(f"[{self.name}] received 2D ROI: {roi_2d}")
        elif rgb_image is not None:
            roi_2d = [0, rgb_image.shape[1], 0, rgb_image.shape[0]]
            self.logger.info(f"[{self.name}] 2D ROI not exist, using full image")
        else:
            roi_2d = []
            self.logger.warning(f"[{self.name}] 2D ROI not exist and rgb_image is None")

        self.roi_3d = []
        return roi_2d, True


def start_grpc_server(mode_info, model, model_list, gpu_id, logger_path):
    MAX_MESSAGE_LENGTH = 2000000000
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
    cmes_ai_pb2_grpc.add_CmesAiExecutorServicer_to_server(
        grpc_runner(mode_info, model, model_list, gpu_id, logger_path), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

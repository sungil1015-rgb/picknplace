import os
os.environ["PYTORCH_JIT"] = "0"
import grpc
import runner.modes.protos.cmes_ai_pb2 as cmes_ai_pb2
import runner.modes.protos.cmes_ai_pb2_grpc as cmes_ai_pb2_grpc
from runner.modes.utils.message_manager import JsonEncoder, JsonDecoder

channel = grpc.insecure_channel('localhost:50051')
client = cmes_ai_pb2_grpc.CmesAiExecutorStub(channel)


def detec_picknplace():
    json_encoder = JsonEncoder()

    json_encoder.set_data('model', 'PickNPlace')
    # json_encoder.set_data('level', 'DEBUG')

    # ── file IO 모드 ──
    # 서버(grpc_mode)가 ./img/ 폴더에서 이미지를 직접 읽는다.
    # 호출 전에 아래 파일을 준비해 둘 것:
    #   ./img/rgb.png      (BGR, uint8)
    #   ./img/depth.png    (16-bit, uint16, mm 단위)
    #   ./img/normal.bin   (header int32 h,w,c + float32 raw)

    json_encoder.set_data('save_result', 1)
    json_encoder.set_data('vis_pcd', 0)
    json_encoder.set_data('save_ply', 0)
    encoded_data = json_encoder.serialize()

    print("end the json encode")
    return bytes(encoded_data, 'utf-8')


def is_connected():
    json_encoder = JsonEncoder()
    json_encoder.set_data('temp', 'temp')
    encoded_data = json_encoder.serialize()
    return bytes(encoded_data, 'utf-8')


def run(function_name):
    if function_name == 'agnostic':
        encoded_data = detec_picknplace()
    elif function_name == 'is_connected':
        encoded_data = is_connected()
    else:
        raise ValueError(f'NOT SUPPORTED: {function_name}')

    response = client.Excute(
        cmes_ai_pb2.Request(request_function_name=function_name, request_data=encoded_data)
    )
    print(f"reply from: {response.reply_function_name}")

    json_decoder = JsonDecoder()
    json_decoder.parseString(response.reply_data)

    # ── 공통: state 출력 ──
    state = json_decoder.get_data('state')
    print(f"state: {state}")

    # ── agnostic 응답일 때만 추론 결과 출력 ──
    if function_name == 'agnostic':
        print(f"  result_data    : {json_decoder.get_data('result_data')}")
        print(f"  class_id       : {json_decoder.get_data('class_id')}")
        print(f"  pts_per_object : {json_decoder.get_data('pts_per_object')}")
        print(f"  suction_points : {json_decoder.get_data('suction_points')}")
        print(f"  2d_roi         : {json_decoder.get_data('2d_roi')}")
        print(f"  3d_roi         : {json_decoder.get_data('3d_roi')}")



if __name__ == '__main__':
    # is_connected : gRPC 통신 확인
    # agnostic     : 추론 파이프라인 전체 확인 (./img/ 에 데이터 필요)
    run('agnostic')

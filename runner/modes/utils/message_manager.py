import numpy as np
import cv2
import json
import base64
import open3d as o3d
import ctypes

def get_img_type(channel, dtype):
    prefix_num = (channel-1)*8
    if (dtype == np.dtype(np.uint8)):
        return prefix_num+0
    elif (dtype == np.dtype(np.int8)):
        return prefix_num+1
    elif (dtype == np.dtype(np.uint16)):
        return prefix_num+2
    elif (dtype == np.dtype(np.int16)):
        return prefix_num+3
    elif (dtype == np.dtype(np.int32)):
        return prefix_num+4
    elif (dtype == np.dtype(np.float32)):
        return prefix_num+5
    elif (dtype == np.dtype(np.float64)):
        return prefix_num+6

def get_image_dtype(type):
    left_value = type % 8
    if(left_value==0):
        return np.dtype(np.uint8)
    elif(left_value==1):
        return np.dtype(np.int8)
    elif(left_value==2):
        return np.dtype(np.uint16)
    elif(left_value==3):
        return np.dtype(np.int16)
    elif(left_value==4):
        return np.dtype(np.int32)
    elif(left_value==5):
        return np.dtype(np.float32)
    elif(left_value==6):
        return np.dtype(np.float64)

def get_image_chnnel(type):
    left_value = type % 8
    channel = type-left_value
    return (int)(channel/8 +1)

def reshape_from_buffer(row, col, channel, data ,dt):
    buffer = np.frombuffer(data, dt)
    if(channel != 1):
        _data = buffer.reshape(row, col, channel)
        return np.copy(_data)
    else:
        _data = buffer.reshape(row, col)
        return np.copy(_data)

def reshape_from_pointcloud_buffer(height, width, channel,data, dt):
    buffer = np.frombuffer(data, dt)
    _data = buffer.reshape(height*width,3)
    return np.copy(_data)


class JsonDecoder:
    def parseString(self, seializedstring):
        self.root = json.loads(seializedstring)

    def get_img_data(self, key):
        img_data = self.root.get(key)
        row = img_data['row']
        col = img_data['col']
        type = img_data['type']
        dt = get_image_dtype(type)
        channel  = get_image_chnnel(type)
        raw_data = img_data['raw_data']
        imgdata = base64.b64decode(raw_data)
        return reshape_from_buffer(row,col,channel,imgdata,dt)

    def get_cloud_data(self, key):
        cloud_data = self.root.get(key)
        height = cloud_data['height']
        width = cloud_data['width']
        channel = 3

        point_cloud = o3d.geometry.PointCloud()

        raw_data_points = cloud_data['raw_data_points']
        raw_data_pt = base64.b64decode(raw_data_points)
        points = reshape_from_pointcloud_buffer(height, width, channel,raw_data_pt,np.dtype(np.float32))
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if 'raw_data_normals' in cloud_data:
            raw_data_normal = cloud_data['raw_data_normals']
            raw_data_nor = base64.b64decode(raw_data_normal)
            normals = reshape_from_pointcloud_buffer(height, width, channel, raw_data_nor,np.dtype(np.float32))
            point_cloud.normals = o3d.utility.Vector3dVector(normals)
        if 'raw_data_rgbs' in cloud_data:
            raw_data_rgbs = cloud_data['raw_data_rgbs']
            raw_data_rgb = base64.b64decode(raw_data_rgbs)
            rgb = reshape_from_pointcloud_buffer(height, width, channel, raw_data_rgb, np.dtype(np.uint8))
            point_cloud.colors = o3d.utility.Vector3dVector(rgb)
        return point_cloud

    def get_data(self, key):
        return self.root.get(key)

    def get_array(self, key):
        jsonArray = self.root.get(key)
        array_data = []
        for data in jsonArray:
            array_data.append(data)
        return array_data

class JsonEncoder:
    def __init__(self):
        self.root = {}

    def set_data(self, key, data):
        self.root[key] = data

    def set_array(self, key, array):
        self.root[key] = array

    def set_image(self, key, img):
        channels = img.shape[-1] if img.ndim == 3 else 1
        img_type = get_img_type(channels, img.dtype)
        buffer = img.reshape(img.shape[0] * img.shape[1] * channels)
        string_repr = base64.standard_b64encode(buffer).decode('utf-8')
        image_dat = {'row' : img.shape[0], 'col' : img.shape[1], 'type' : img_type, 'raw_data' : string_repr}
        self.root[key] = image_dat

    #point_cloud must be open3d
    def set_point_cloud(self, key, point_cloud):
        points = np.asarray(point_cloud.points)

        height = 1
        width =points.shape[0]
        channel = points.shape[1]

        pts = points.reshape(height*width*channel).astype(np.float32)
        string_repr_pts = base64.standard_b64encode(pts).decode('utf-8')
        cloud_dat = {'height': height, 'width': width, 'raw_data_points': string_repr_pts}

        normals = np.asarray(point_cloud.normals)
        if normals.shape[0] != 0:
            nmls = normals.reshape(height * width * channel).astype(np.float32)
            string_repr_nmls = base64.standard_b64encode(nmls).decode('utf-8')
            cloud_dat['raw_data_normals'] = string_repr_nmls

        colors = np.asarray(point_cloud.colors)
        if colors.shape[0] != 0:
            cls = colors.reshape(height * width * channel).astype(np.uint8)
            string_repr_cls = base64.standard_b64encode(cls).decode('utf-8')
            cloud_dat['raw_data_rgbs'] = string_repr_cls

        self.root[key] = cloud_dat

    def serialize(self):
        return json.dumps(self.root)







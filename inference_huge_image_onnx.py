import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
from torch import nn

from osgeo import gdal
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import logging
import cv2
from tqdm import tqdm
import numpy as np
from torch import optim
from tqdm import tqdm
from utils.dataset import BuildingDataset
from torch.utils.data import DataLoader
from utils.sync_batchnorm.batchnorm import convert_model

import onnx
import onnxruntime as ort

import logging



def subset_info_filter(ds, subset_info):
    if subset_info[0] > ds.RasterXSize or subset_info[1] > ds.RasterYSize:
        raise Exception('x, y start in subset_info must be inside the raster extent\
            you enter the x,y starts are [ %s, %s ], and the size of the raster is \
                [ %s, %s ]' % (subset_info[0], subset_info[1], ds.RasterXSize, ds.RasterYSize))

    if subset_info[0] + subset_info[2] > ds.RasterXSize:
        subset_info[2] = ds.RasterXSize - subset_info[0]
        
    if subset_info[1] + subset_info[3] > ds.RasterYSize:
        subset_info[3] = ds.RasterYSize - subset_info[1]
    
    return subset_info

SCALE_FLAG = {
    'INTER_NEAREST' : cv2.INTER_NEAREST,
    'INTER_LINEAR' : cv2.INTER_LINEAR,
    'INTER_AREA' : cv2.INTER_AREA,
    'INTER_CUBIC' : cv2.INTER_CUBIC,
    'INTER_LANCZOS4' : cv2.INTER_LANCZOS4,
}

def updateGeoTransform(ds, subset_info=None, scale=None):
    '''update GeoTransform of gdal.Datasource'''
    gt = list(ds.GetGeoTransform())

    if scale is not None:
        gt[1] /= scale
        gt[5] /= scale
    else:
        scale = 1

    
    if subset_info is not None:
        gt[0] += subset_info[0] * gt[1] * scale
        gt[3] += subset_info[1]* gt[5] * scale
    
    ds.SetGeoTransform(gt)
    return ds


def loadGeoRaster(src_dir:str, scale:float=None, return_ds=False,
            subset_info:list=None, single_band:int=None, scale_method:str=None):
    '''load entire or part of raster data as array
    
    Args:
        -src_dir: the directory of raster file, support file format:
            -- TIFF
            -- IMG
            TODO the file format has sub-datasource (hd5)
        
        -scale [optional]: the factor to zoom raster data, now support fixed scale,
        which means the size of loaded is divided scale by original size
        
        -subset_info [optional]: a four-elements list like 
                        [start_x, start_y, offset_x, offset_y]
        
        -single_band [optional]: load only bands from multi-bands raster

        -scale_method [optional]: interplote method, including 'INTER_NEAREST',\
                'INTER_LINEAR','INTER_AREA','INTER_CUBIC','INTER_LANCZOS4', \
                default is INTER_NEAREST

    Returns:
        loaded_arr: A ndarray with shape as [band, height, width]
        
        gdal.datasource: a fixed gdal.datasource according to scale and 
                            subset_info

    Notes:
        1. It is stipilated here that
            gdal.datasource.RasterXsize -> width -> ndarray.shape[-1] -> gt[0] -> subset_info[0]
            gdal.datasource.RasterYsize -> height-> ndarray.shape[-2] -> gt[3] -> subset_info[0]
        2. In the laoding process, subset first and then scale
        3. single_band start from 1, end to the band number of the raster
    '''

    if not os.path.isfile(src_dir):
        raise FileExistsError(': [%s]' % src_dir)
    
    ds = gdal.Open(src_dir, 1) # mode 0 -> readonly, 1 -> writeble
    # print(ds.GetGeoTransform())
    # TODO if subset_info overflow
    subset_info = [0, 0, ds.RasterXSize, ds.RasterYSize] if \
        subset_info is None else subset_info_filter(ds, subset_info)

    # print(subset_info)

    if isinstance(single_band, int) and single_band>0 and single_band<=ds.RasterCount:
        raster_array = ds.GetRasterBand(single_band).ReadAsArray(*subset_info)
    else:
        raster_array = ds.ReadAsArray(*subset_info)
    
    if isinstance(scale, float):
        scale_method = SCALE_FLAG[scale_method] if scale_method is not None else cv2.INTER_LINEAR
        raster_array = cv2.resize(
                raster_array.transpose(1,2,0), 
                (0,0), 
                fx=scale, 
                fy=scale, 
                interpolation=scale_method
            ).transpose(2,0,1)

    if subset_info or scale:
        ds = updateGeoTransform(ds, subset_info=subset_info, scale=scale)
    # print(ds.GetGeoTransform())
    if return_ds:
        return raster_array, ds
    else:
        return raster_array


def getClipMap(im_shape, patch_size, buffer_size):
    w, h = im_shape[-2:]
    clip_map = []
    stride = patch_size - 2* buffer_size
    
    n_x = w // stride if w % stride == 0 else (w // stride +1)
    n_y = h // stride if h % stride == 0 else (h // stride +1)

    '''add buffer to top and left'''
    for xi in range(n_x):
        for yi in range(n_y):
            clip_map.append([xi, yi])

    return clip_map


def inference(patch, model, device):

    predict_transform = A.Compose(
        [
           
            # A.CenterCrop(img_size),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Normalize([0.34728308,0.48111935,0.35607836], [0.16713127, 0.17759323, 0.15076649]),
            ToTensorV2()
        ]
    )

    augmentations = predict_transform(image = patch)
    patch = augmentations["image"]

    patch = torch.unsqueeze(patch,dim=0)
    predict_loader = DataLoader(
        patch,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle =False,
    )
    
    model.eval()

    for idx,imgs in enumerate(predict_loader):
        imgs = imgs.to(device = device, dtype=torch.float32)

        with torch.no_grad():

            pred = model(imgs)

            mask_pred = pred[9].detach().cpu().numpy().squeeze()
            edge_pred = pred[11].detach().cpu().numpy().squeeze()
            dist_pred = pred[10].detach().cpu().numpy().squeeze()


            mask_pred = torch.from_numpy(mask_pred)
            mask_pred = torch.sigmoid(mask_pred)
            threshold = 0.62        #0.63
            mask_bin = np.where(mask_pred > threshold, 2, 0) # 阈值化并转换为浮点数
            mask_bin = np.array(mask_bin, dtype='uint8')

            edge_pred = torch.from_numpy(edge_pred)
            edge_pred = torch.sigmoid(edge_pred)
            threshold = 0.37          #0.39
            edge_bin = np.where(edge_pred > threshold, 1, 0) # 阈值化并转换为浮点数
            edge_bin = np.array(edge_bin, dtype='uint8')

            # boundary_pixels = np.array(edge_bin) == 1
            # mask_bin[boundary_pixels] = 1


            result = edge_bin

    return result


def inferenceEntireImage(im, model, patch_size, buffer_size, device):
    w, h = im.shape[-2:]
    clip_map = getClipMap(im.shape, patch_size, buffer_size)
    
    out_im = np.zeros(im.shape[-2:], dtype=np.uint8)

    im = np.pad(im, ((0, 0),(buffer_size, buffer_size),(buffer_size,buffer_size )), 'edge')

    stride = patch_size - 2* buffer_size
    all_image = len(clip_map)
    flag = 1
    for xi, yi in tqdm(clip_map):
        
        xs = xi*stride
        xe = xi*stride+patch_size
        
        ys = yi*stride
        ye = yi*stride+patch_size
        
        pad_x = 0
        pad_y = 0
    
        if xe > w+2*buffer_size:
            pad_x = xe - w-2*buffer_size
            xe = w+2*buffer_size
            
        if ye > h+2*buffer_size:
            pad_y = ye -h-2*buffer_size
            ye = h+2*buffer_size
            

        patch = im[:, xs: xe, ys: ye]
        if pad_x or pad_y:
            patch = np.pad(patch, ((0, 0),(0, pad_x),(0, pad_y)), 'reflect')

        patch = patch.transpose(1,2,0)

        pred_patch = inference(patch, model, device)
        # print(str(flag)+"/"+str(all_image))
        flag += 1
        if len(np.unique(patch)) != 1:
            out_im[xs: xs+stride-pad_x, ys: ys+stride-pad_y] = pred_patch[
            buffer_size: buffer_size+stride-pad_x, buffer_size: buffer_size+stride-pad_y]
            
    return out_im  


def load_checkpoint(checkpoint,model):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(checkpoint))


def reclass(image):
    image[image == 1] = 0
    image[image == 2] = 1
    return image


def WriteTiff(img_arr, dst_dir, ds=None, dst_fileformat = 'tif',
    size_source='in_arr', dtype = None):

    pos=dst_dir.rfind('.')
    if pos == -1:
        dst_fileformat = 'envi'
    else:
        dst_fileformat = dst_dir[pos+1:]

    dst_fileformat = dst_fileformat.strip().upper()

    assert dst_fileformat in ['TIFF', 'TIF', 'PNG','JPG','JPEG','ENVI'], 'please check your dst_dir and change the right fileformat'

    io_dtype = {
        gdal.GDT_Byte: np.dtype(np.uint8),
        gdal.GDT_UInt16: np.dtype(np.uint16),
        gdal.GDT_Float32: np.dtype(np.float32),
    }
    
    oi_dtype = {v:k for k, v in io_dtype.items()}

    dtype = dtype if dtype is not None else \
        oi_dtype[ np.dtype(img_arr.dtype) ]

    driver = gdal.GetDriverByName('MEM')
    if len(img_arr.shape) == 3:
        nb = img_arr.shape[0]
    elif len(img_arr.shape) == 2:
        nb = 1
    else:
        raise ValueError('The size of input tensor has abnormal size in %s.'
            % img_arr.shape)
    

    if size_source == 'in_arr':
        outRaster = driver.Create(
            dst_dir, img_arr.shape[-1], img_arr.shape[-2], nb, dtype)
    elif size_source == 'ref_ds':
        outRaster = driver.Create(
            dst_dir, img_arr.shape[-1], img_arr.shape[-2], nb, dtype)

    if ds is not None:
        outRaster.SetGeoTransform(ds.GetGeoTransform())
        outRaster.SetProjection(ds.GetProjection())

    if dtype is not None and img_arr.dtype != io_dtype[dtype]:
        img_arr = img_arr.astype(io_dtype[dtype])
    

    if nb == 1:
        outRaster.GetRasterBand(1).WriteArray(img_arr)
    elif nb > 1:
        for band_idx in range(nb):
            outRaster.GetRasterBand(band_idx + 1).WriteArray(img_arr[band_idx,:,:])

    if dst_fileformat in ['JPG','JPEG']:
        driver_out = gdal.GetDriverByName('JPEG')
    elif dst_fileformat == 'PNG':
        driver_out = gdal.GetDriverByName('PNG')
    elif dst_fileformat in ['TIFF', 'TIF']:
        driver_out = gdal.GetDriverByName('GTIFF')
    # elif dst_fileformat == 'ENVI':
    #     driver_out = gdal.GetDriverByName('ENVI')

    outImage = driver_out.CreateCopy(dst_dir,outRaster)
    outRaster.FlushCache()
    # outImage.FlushCache()


def predict(predict_img,model,tif_dir,shp_dir,temp_path,device):
        device = device
        # predict
        im, ds = loadGeoRaster(predict_img,return_ds=True)
        if im.shape[0] > 3:
            im = im[0:3,:,:]
        w,h = im[-2:]
        w = w.shape[1]
        h = h.shape[0]


        print('=> predicting')
        out_im = inferenceEntireImage(im,model,512,128,device = device)

        ### raster posses
        print('=> raster post-processing')
        mat = out_im.astype(np.uint8)
        # mat = reclass(mat)
        # mat = get_outline(mat,w,h)
        # mat = expand(mat,1,1,w,h)
        # mat = delete(mat,w,h)
        WriteTiff(mat,tif_dir,ds=ds)

def batch_predict(image_dir, output_dir, net, device):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 列出影像目录中的所有.tif文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    # 使用tqdm显示进度条
    for image_file in tqdm(image_files, desc="Processing images"):
        PREDICT_IMG = os.path.join(image_dir, image_file)
        base_filename = os.path.splitext(image_file)[0]
        TIF_DIR = os.path.join(output_dir, f"{base_filename}_result.tif")
        SHP_DIR = os.path.join(output_dir, f"{base_filename}_result.shp")
        TEMP_PATH = os.path.join(output_dir, 'temp')
        
        # 确保临时目录存在
        if not os.path.exists(TEMP_PATH):
            os.makedirs(TEMP_PATH)
        
        # 调用预测函数
        predict(PREDICT_IMG, net, TIF_DIR, SHP_DIR, TEMP_PATH, device)

def convert_pth_to_onnx(model, model_name, checkpoint_dir, onnx_file_path, device):
    model_path = os.path.join(checkpoint_dir, model_name + '.pth')
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()


    dummy_input = torch.randn(1, 3, 512, 512).to(device)  # 假设输入是512x512的三通道图像

    # 将模型转换为ONNX格式
    torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True,
                      opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    logging.info(f'Model converted to ONNX format and saved as {onnx_file_path}')

def batch_predict(image_dir, output_dir, onnx_model_path, device):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载ONNX模型
    ort_session = ort.InferenceSession(onnx_model_path)

    # 列出影像目录中的所有.tif文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    # 使用tqdm显示进度条
    for image_file in tqdm(image_files, desc="Processing images"):
        PREDICT_IMG = os.path.join(image_dir, image_file)
        base_filename = os.path.splitext(image_file)[0]
        TIF_DIR = os.path.join(output_dir, f"{base_filename}_result.tif")
        SHP_DIR = os.path.join(output_dir, f"{base_filename}_result.shp")
        TEMP_PATH = os.path.join(output_dir, 'temp')
        
        # 确保临时目录存在
        if not os.path.exists(TEMP_PATH):
            os.makedirs(TEMP_PATH)
        
        # 调用ONNX模型进行预测
        predict_onnx(PREDICT_IMG, TIF_DIR, SHP_DIR, TEMP_PATH, ort_session)

def predict_onnx(image_path, tif_output_path, shp_output_path, temp_path, ort_session):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))  # 根据需要调整大小
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # 转换为CHW格式
    image = np.expand_dims(image, axis=0)  # 添加批次维度

    # ONNX推理
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)

    # 假设模型输出是需要处理的掩码
    output_mask = ort_outs[0].squeeze()
    
    # 保存结果
    cv2.imwrite(tif_output_path, (output_mask * 255).astype(np.uint8))

    # 如果有其他处理（如保存.shp文件），在此处添加



if __name__ == "__main__":

    # ***************************************************
    batchsize = 1
    num_workers = 24
    read_name = 'test_best'
    Dataset = 'UAV'
    net = DCP()

    # data_dir = "/home3/lxy/cul_plot/EMLT/dataset_UAV/"
    dir_checkpoint = 'weights/DCP/'
    onnx_file_path = './onnx/DCP.onnx'  # 保存ONNX文件的路径

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    # 将.pth模型转换为ONNX格式
    convert_pth_to_onnx(net, read_name, dir_checkpoint, onnx_file_path, device)
    print("convert_pth_to_onnx sucessful")
    


    image_dir = '..../full_image'
    output_dir = '..../mask_bd_t_full_image'
    # 批处理所有影像
    batch_predict(image_dir, output_dir, net, device)

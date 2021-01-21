from dataclasses import dataclass, fields, field
import numpy as np
import gf
import pkg_db
import os
from PIL import Image
import binascii
from typing import List
import texture2ddecoder

"""
Images are a two-part system. The first file is the image header, containing all the important info. The second part
has the actual image data which uses the header data to transcribe that data to an actual image.
"""


@dataclass
class ImageHeader:
    TargetSize: np.uint32 = np.uint32(0)  # 0
    TextureFormat: np.uint32 = np.uint32(0)  # 4
    Field8: np.uint32 = np.uint32(0)  # 8
    FieldC:  np.uint32 = np.uint32(0)  # C
    Field10: np.uint32 = np.uint32(0)  # 10
    Field14: np.uint32 = np.uint32(0)  # 14
    Field18: np.uint32 = np.uint32(0)  # 18
    Field1C: np.uint32 = np.uint32(0)  # 1C
    Cafe: np.uint16 = np.uint16(0)  # 20  0xCAFE
    Width: np.uint16 = np.uint16(0)  # 22
    Height: np.uint16 = np.uint16(0)  # 24
    Field26: np.uint16 = np.uint16(0)
    TA: np.uint16 = np.uint16(0)  # 28
    Field2A: np.uint16 = np.uint16(0)
    Field2C: np.uint32 = np.uint32(0)
    Field30: np.uint32 = np.uint32(0)
    Field34: np.uint32 = np.uint32(0)
    Field38: np.uint32 = np.uint32(0)
    LargeTextureHash: np.uint32 = np.uint32(0)  # 3C
    TextureFormatDefined: str = ''

# This header includes the magic number, DDS header, and DXT10 DDS header
@dataclass
class DDSHeader:
    MagicNumber: np.uint32 = np.uint32(0)
    dwSize: np.uint32 = np.uint32(0)
    dwFlags: np.uint32 = np.uint32(0)
    dwHeight: np.uint32 = np.uint32(0)
    dwWidth: np.uint32 = np.uint32(0)
    dwPitchOrLinearSize: np.uint32 = np.uint32(0)
    dwDepth: np.uint32 = np.uint32(0)
    dwMipMapCount: np.uint32 = np.uint32(0)
    dwReserved1: List[np.uint32] = field(default_factory=list)  # size 11, [11]
    dwPFSize: np.uint32 = np.uint32(0)
    dwPFFlags: np.uint32 = np.uint32(0)
    dwPFFourCC: np.uint32 = np.uint32(0)
    dwPFRGBBitCount: np.uint32 = np.uint32(0)
    dwPFRBitMask: np.uint32 = np.uint32(0)
    dwPFGBitMask: np.uint32 = np.uint32(0)
    dwPFBBitMask: np.uint32 = np.uint32(0)
    dwPFABitMask: np.uint32 = np.uint32(0)
    dwCaps: np.uint32 = np.uint32(0)
    dwCaps2: np.uint32 = np.uint32(0)
    dwCaps3: np.uint32 = np.uint32(0)
    dwCaps4: np.uint32 = np.uint32(0)
    dwReserved2: np.uint32 = np.uint32(0)
    dxgiFormat: np.uint32 = np.uint32(0)
    resourceDimension: np.uint32 = np.uint32(0)
    miscFlag: np.uint32 = np.uint32(0)
    arraySize: np.uint32 = np.uint32(0)
    miscFlags2: np.uint32 = np.uint32(0)


DXGI_FORMAT = [
  "DXGI_FORMAT_UNKNOWN",
  "DXGI_FORMAT_R32G32B32A32_TYPELESS",
  "DXGI_FORMAT_R32G32B32A32_FLOAT",
  "DXGI_FORMAT_R32G32B32A32_UINT",
  "DXGI_FORMAT_R32G32B32A32_SINT",
  "DXGI_FORMAT_R32G32B32_TYPELESS",
  "DXGI_FORMAT_R32G32B32_FLOAT",
  "DXGI_FORMAT_R32G32B32_UINT",
  "DXGI_FORMAT_R32G32B32_SINT",
  "DXGI_FORMAT_R16G16B16A16_TYPELESS",
  "DXGI_FORMAT_R16G16B16A16_FLOAT",
  "DXGI_FORMAT_R16G16B16A16_UNORM",
  "DXGI_FORMAT_R16G16B16A16_UINT",
  "DXGI_FORMAT_R16G16B16A16_SNORM",
  "DXGI_FORMAT_R16G16B16A16_SINT",
  "DXGI_FORMAT_R32G32_TYPELESS",
  "DXGI_FORMAT_R32G32_FLOAT",
  "DXGI_FORMAT_R32G32_UINT",
  "DXGI_FORMAT_R32G32_SINT",
  "DXGI_FORMAT_R32G8X24_TYPELESS",
  "DXGI_FORMAT_D32_FLOAT_S8X24_UINT",
  "DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS",
  "DXGI_FORMAT_X32_TYPELESS_G8X24_UINT",
  "DXGI_FORMAT_R10G10B10A2_TYPELESS",
  "DXGI_FORMAT_R10G10B10A2_UNORM",
  "DXGI_FORMAT_R10G10B10A2_UINT",
  "DXGI_FORMAT_R11G11B10_FLOAT",
  "DXGI_FORMAT_R8G8B8A8_TYPELESS",
  "DXGI_FORMAT_R8G8B8A8_UNORM",
  "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB",
  "DXGI_FORMAT_R8G8B8A8_UINT",
  "DXGI_FORMAT_R8G8B8A8_SNORM",
  "DXGI_FORMAT_R8G8B8A8_SINT",
  "DXGI_FORMAT_R16G16_TYPELESS",
  "DXGI_FORMAT_R16G16_FLOAT",
  "DXGI_FORMAT_R16G16_UNORM",
  "DXGI_FORMAT_R16G16_UINT",
  "DXGI_FORMAT_R16G16_SNORM",
  "DXGI_FORMAT_R16G16_SINT",
  "DXGI_FORMAT_R32_TYPELESS",
  "DXGI_FORMAT_D32_FLOAT",
  "DXGI_FORMAT_R32_FLOAT",
  "DXGI_FORMAT_R32_UINT",
  "DXGI_FORMAT_R32_SINT",
  "DXGI_FORMAT_R24G8_TYPELESS",
  "DXGI_FORMAT_D24_UNORM_S8_UINT",
  "DXGI_FORMAT_R24_UNORM_X8_TYPELESS",
  "DXGI_FORMAT_X24_TYPELESS_G8_UINT",
  "DXGI_FORMAT_R8G8_TYPELESS",
  "DXGI_FORMAT_R8G8_UNORM",
  "DXGI_FORMAT_R8G8_UINT",
  "DXGI_FORMAT_R8G8_SNORM",
  "DXGI_FORMAT_R8G8_SINT",
  "DXGI_FORMAT_R16_TYPELESS",
  "DXGI_FORMAT_R16_FLOAT",
  "DXGI_FORMAT_D16_UNORM",
  "DXGI_FORMAT_R16_UNORM",
  "DXGI_FORMAT_R16_UINT",
  "DXGI_FORMAT_R16_SNORM",
  "DXGI_FORMAT_R16_SINT",
  "DXGI_FORMAT_R8_TYPELESS",
  "DXGI_FORMAT_R8_UNORM",
  "DXGI_FORMAT_R8_UINT",
  "DXGI_FORMAT_R8_SNORM",
  "DXGI_FORMAT_R8_SINT",
  "DXGI_FORMAT_A8_UNORM",
  "DXGI_FORMAT_R1_UNORM",
  "DXGI_FORMAT_R9G9B9E5_SHAREDEXP",
  "DXGI_FORMAT_R8G8_B8G8_UNORM",
  "DXGI_FORMAT_G8R8_G8B8_UNORM",
  "DXGI_FORMAT_BC1_TYPELESS",
  "DXGI_FORMAT_BC1_UNORM",
  "DXGI_FORMAT_BC1_UNORM_SRGB",
  "DXGI_FORMAT_BC2_TYPELESS",
  "DXGI_FORMAT_BC2_UNORM",
  "DXGI_FORMAT_BC2_UNORM_SRGB",
  "DXGI_FORMAT_BC3_TYPELESS",
  "DXGI_FORMAT_BC3_UNORM",
  "DXGI_FORMAT_BC3_UNORM_SRGB",
  "DXGI_FORMAT_BC4_TYPELESS",
  "DXGI_FORMAT_BC4_UNORM",
  "DXGI_FORMAT_BC4_SNORM",
  "DXGI_FORMAT_BC5_TYPELESS",
  "DXGI_FORMAT_BC5_UNORM",
  "DXGI_FORMAT_BC5_SNORM",
  "DXGI_FORMAT_B5G6R5_UNORM",
  "DXGI_FORMAT_B5G5R5A1_UNORM",
  "DXGI_FORMAT_B8G8R8A8_UNORM",
  "DXGI_FORMAT_B8G8R8X8_UNORM",
  "DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM",
  "DXGI_FORMAT_B8G8R8A8_TYPELESS",
  "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB",
  "DXGI_FORMAT_B8G8R8X8_TYPELESS",
  "DXGI_FORMAT_B8G8R8X8_UNORM_SRGB",
  "DXGI_FORMAT_BC6H_TYPELESS",
  "DXGI_FORMAT_BC6H_UF16",
  "DXGI_FORMAT_BC6H_SF16",
  "DXGI_FORMAT_BC7_TYPELESS",
  "DXGI_FORMAT_BC7_UNORM",
  "DXGI_FORMAT_BC7_UNORM_SRGB",
  "DXGI_FORMAT_AYUV",
  "DXGI_FORMAT_Y410",
  "DXGI_FORMAT_Y416",
  "DXGI_FORMAT_NV12",
  "DXGI_FORMAT_P010",
  "DXGI_FORMAT_P016",
  "DXGI_FORMAT_420_OPAQUE",
  "DXGI_FORMAT_YUY2",
  "DXGI_FORMAT_Y210",
  "DXGI_FORMAT_Y216",
  "DXGI_FORMAT_NV11",
  "DXGI_FORMAT_AI44",
  "DXGI_FORMAT_IA44",
  "DXGI_FORMAT_P8",
  "DXGI_FORMAT_A8P8",
  "DXGI_FORMAT_B4G4R4A4_UNORM",
  "DXGI_FORMAT_P208",
  "DXGI_FORMAT_V208",
  "DXGI_FORMAT_V408",
  "DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE",
  "DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE",
  "DXGI_FORMAT_FORCE_UINT"
]

def get_header(file_hex):
    img_header = ImageHeader()
    for f in fields(img_header):
        if f.type == np.uint32:
            flipped = "".join(gf.get_flipped_hex(file_hex, 8))
            value = np.uint32(int(flipped, 16))
            setattr(img_header, f.name, value)
            file_hex = file_hex[8:]
        elif f.type == np.uint16:
            flipped = "".join(gf.get_flipped_hex(file_hex, 4))
            value = np.uint16(int(flipped, 16))
            setattr(img_header, f.name, value)
            file_hex = file_hex[4:]
    return img_header


def define_texture_format(texture_format):
    if DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_BC1_UNORM':
        return 'BC1'
    elif DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_BC1_UNORM_SRGB':
        return 'BC1_SRGB'
    elif DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_BC7_UNORM':
        return 'BC7'
    elif DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_BC7_UNORM_SRGB':
        return 'BC7_SRGB'
    elif DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_R8G8B8A8_UNORM':
        return 'RGBA'
    elif DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_R8G8B8A8_UNORM_SRGB':
        return 'RGBA_SRGB'
    elif DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_BC3_UNORM_SRGB':
        return 'BC3_SRGB'
    elif DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_BC4_UNORM':
        return 'BC4'
    elif DXGI_FORMAT[texture_format] == 'DXGI_FORMAT_BC5_UNORM':
        return 'BC5'
    else:
        return DXGI_FORMAT[texture_format]


def get_image_from_file(file_path, all_file_info, save_path=None):
    file_name = file_path.split('/')[-1].split('.')[0]
    file_pkg = file_path.split('/')[-2]
    # To get the actual image data we need to pull this specific file's data from the database as it references its file
    # in its RefID.
    # TODO FIX THIS REMOVE IT BAD BAD BAD
    ref_file = f"{all_file_info[file_name]['RefPKG'][2:]}-{all_file_info[file_name]['RefID'][2:]}"
    ref_pkg = gf.get_pkg_name(ref_file)
    if all_file_info[file_name]['FileType'] == 'Texture Header':
        header_hex = gf.get_hex_data(file_path)
        data_hex = gf.get_hex_data(f'I:/d2_output_3_0_2_0/{ref_pkg}/{ref_file}.bin')
    elif all_file_info[file_name]['FileType'] == 'Texture Data':
        print('Only pass through header please, cba to fix this.')
        return
    else:
        print(f"File given is not texture data or header of type {all_file_info[file_name]['FileType']}")
        return
    header = get_header(header_hex)
    dimensions = [header.Width, header.Height]
    header.TextureFormatDefined = define_texture_format(header.TextureFormat)
    large_tex_hash = gf.get_flipped_hex(hex(header.LargeTextureHash)[2:].upper(), 8)
    # print(large_tex_hash)
    if large_tex_hash != 'FFFFFFFF':
        large_file = gf.get_file_from_hash(large_tex_hash)
        pkg_name = gf.get_pkg_name(large_file)
        data_hex = gf.get_hex_data(f'I:/d2_output_3_0_2_0/{pkg_name}/{large_file}.bin')
    print(ref_file)
    img = get_image_from_data(header, dimensions, data_hex)
    if img:
        if save_path:
            img.save(f'{save_path}/{file_name}.png')
        # else:
        #     img.save(f'C:/d2_output_2_9_2_0_images/{file_pkg}/{file_name}.png')
        #     img.show()


def get_image_from_data(header, dimensions, data_hex):
    img = None
    if 'RGBA' in header.TextureFormatDefined:
        try:
            img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex))
        except ValueError:
            return 'Invalid'
    elif 'BC1' in header.TextureFormatDefined:
        try:
            img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex), 'bcn', (1,))
        except ValueError:
            # bc1_decomp(header, data_hex)
            try:
                dec = texture2ddecoder.decode_bc1(bytes.fromhex(data_hex), header.Width, header.Height)
                img = Image.frombytes('RGBA', dimensions, dec, 'raw', ("BGRA"))
            except ValueError:
                return 'Invalid'
    elif 'BC3' in header.TextureFormatDefined:
        try:
            img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex), 'bcn', (3,))
        except ValueError:
            try:
                dec = texture2ddecoder.decode_bc3(bytes.fromhex(data_hex), header.Width, header.Height)
                img = Image.frombytes('RGBA', dimensions, dec, 'raw', ("BGRA"))
            except ValueError:
                return 'Invalid'
    elif 'BC4' in header.TextureFormatDefined:
        try:
            img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex), 'bcn', (4,))
        except ValueError:
            try:
                dec = texture2ddecoder.decode_bc4(bytes.fromhex(data_hex), header.Width, header.Height)
                img = Image.frombytes('RGBA', dimensions, dec, 'raw', ("BGRA"))
            except ValueError:
                return 'Invalid'
    elif 'BC5' in header.TextureFormatDefined:
        try:
            img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex), 'bcn', (5,))
        except ValueError:
            try:
                dec = texture2ddecoder.decode_bc5(bytes.fromhex(data_hex), header.Width, header.Height)
                img = Image.frombytes('RGBA', dimensions, dec, 'raw', ("BGRA"))
            except ValueError:
                return 'Invalid'
    elif 'BC6' in header.TextureFormatDefined:
        # bc6h_decomp(header, data_hex)
        try:
            img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex), 'bcn', (6,))
        except ValueError:
            try:
                dec = texture2ddecoder.decode_bc6(bytes.fromhex(data_hex), header.Width, header.Height)
                img = Image.frombytes('RGBA', dimensions, dec, 'raw', ("BGRA"))
            except ValueError:
                return 'Invalid'
    elif 'BC7' in header.TextureFormatDefined:
        # print('Size', int(len(data_hex)/2), '|', header.TextureFormatDefined, dimensions)
        try:
            img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex), 'bcn', (7,))
        except ValueError:
            # bc7_decomp(header, data_hex)
            try:
                dec = texture2ddecoder.decode_bc7(bytes.fromhex(data_hex), header.Width, header.Height)
                img = Image.frombytes('RGBA', dimensions, dec, 'raw', ("BGRA"))
            except ValueError:
                return 'Invalid'
    else:
        print(f'Image not supported type {header.TextureFormatDefined}')
    return img


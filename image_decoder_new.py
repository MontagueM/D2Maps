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
    Field0: np.uint32 = np.uint32(0)  # 0
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
    Field28: np.uint32 = np.uint32(0)
    Field2C: np.uint32 = np.uint32(0)
    Field30: np.uint32 = np.uint32(0)
    Field34: np.uint32 = np.uint32(0)
    Field38: np.uint32 = np.uint32(0)
    LargeTextureHash: np.uint32 = np.uint32(0)  # 3C

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


with open('dxgi.format') as f:
    DXGI_FORMAT = f.readlines()


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

    # DXGI_FORMAT_BC6H_UF16 is HDR, don't think I'm able to extract easily
    # Some other unsupported formats: DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_BC5_UNORM


def get_image_from_file(file_path, save_path=None):
    pkg_db.start_db_connection()
    file_name = file_path.split('/')[-1].split('.')[0]
    file_pkg = file_path.split('/')[-2]
    # To get the actual image data we need to pull this specific file's data from the database as it references its file
    # in its RefID.
    # TODO FIX THIS REMOVE IT BAD BAD BAD
    entries = pkg_db.get_entries_from_table(file_pkg, 'FileName, RefID, RefPKG, FileType')
    this_entry = [x for x in entries if x[0] == file_name][0]
    ref_file_name = f'{this_entry[2][2:]}-{gf.fill_hex_with_zeros(this_entry[1][2:], 4)}'
    ref_pkg = gf.get_pkg_name(ref_file_name)
    if this_entry[-1] == 'Texture Header':
        header_hex = gf.get_hex_data(file_path)
        data_hex = gf.get_hex_data(f'I:/d2_output_3_0_0_4/{ref_pkg}/{ref_file_name}.bin')
    elif this_entry[-1] == 'Texture Data':
        print('Only pass through header please, cba to fix this.')
        return
    else:
        print(f"File given is not texture data or header of type {this_entry[-1]}")
        return
    header = get_header(header_hex)
    dimensions = [header.Width, header.Height]
    large_tex_hash = gf.get_flipped_hex(hex(header.LargeTextureHash)[2:].upper(), 8)
    # print(large_tex_hash)
    if large_tex_hash != 'FFFFFFFF':
        large_file = gf.get_file_from_hash(large_tex_hash)
        pkg_name = gf.get_pkg_name(large_file)
        data_hex = gf.get_hex_data(f'I:/d2_output_3_0_0_4/{pkg_name}/{large_file}.bin')
    print(file_name, ref_file_name)
    img = get_image_from_data(header, dimensions, data_hex, save_path)
    if img:
        if save_path:
            img.save(f'{save_path}/{file_name}.png')
        else:
            img.save(f'I:/d2_output_3_0_0_2_images/{file_pkg}/{file_name}.png')
            img.show()


def bc_decomp(header, data_hex, save_path):
    """
    8 bytes per 4x4 pixel block
    dwFourCC "DXT1"
    dwFlags DDS_FOURCC
    """
    # width_hex = gf.get_flipped_hex(gf.fill_hex_with_zeros(hex(header.Width)[2:], 8), 8).upper()
    # height_hex = gf.get_flipped_hex(gf.fill_hex_with_zeros(hex(header.Height)[2:], 8), 8).upper()
    block_size = 16
    # dw_pitch_or_linear_size = np.uint32(max(1, ((header.Width+3) / 4) * block_size))
    # dw_hex = gf.get_flipped_hex(gf.fill_hex_with_zeros(hex(int(dw_pitch_or_linear_size))[2:], 8), 8).upper()
    ######
    # data_size = int(len(data_hex)/2)
    # c_data_size = int(data_size/2)  # c_ is since its BC7 compressed, it is stored with half the size for normal


    bc1_header = DDSHeader()
    bc1_header.MagicNumber = int('20534444', 16)
    bc1_header.dwSize = 124
    bc1_header.dwFlags = (0x1 + 0x2 + 0x4 + 0x1000) + 0x80000
    bc1_header.dwHeight = header.Height
    bc1_header.dwWidth = header.Width
    bc1_header.dwPitchOrLinearSize = max(1, (bc1_header.dwWidth+3)/4)*block_size
    bc1_header.dwDepth = 0
    bc1_header.dwMipMapCount = 0
    bc1_header.dwReserved1 = [0]*11
    bc1_header.dwPFSize = 32
    bc1_header.dwPFFlags = 0x1 + 0x4  # contains alpha data + contains compressed RGB data
    bc1_header.dwPFFourCC = int('30315844', 16)
    bc1_header.dwPFRGBBitCount = 0
    bc1_header.dwPFRBitMask = 0  # All of these are 0 as it is compressed data
    bc1_header.dwPFGBitMask = 0
    bc1_header.dwPFBBitMask = 0
    bc1_header.dwPFABitMask = 0
    # bc1_header.dwCaps = 0x1000 + 0x8
    # bc1_header.dwCaps2 = 0x200 + 0x400 + 0x800 + 0x1000 + 0x2000 + 0x4000 + 0x8000  # All faces for cubemap
    bc1_header.dwCaps = 0x1000
    bc1_header.dwCaps2 = 0
    bc1_header.dwCaps3 = 0
    bc1_header.dwCaps4 = 0
    bc1_header.dwReserved2 = 0
    bc1_header.dxgiFormat = header.TextureFormat
    bc1_header.resourceDimension = 3  # DDS_DIMENSION_TEXTURE2D
    bc1_header.miscFlag = 0
    # int(((int(bc1_header.dwWidth) * int(bc1_header.dwHeight)) + 320) / c_data_size)
    bc1_header.arraySize = 1
    bc1_header.miscFlags2 = 0x1 #?
    print(f'Array size {bc1_header.arraySize}')
    write_file(bc1_header, data_hex, save_path)


def write_file(header, file_hex, save_path):
    with open(save_path, 'wb') as b:
        for f in fields(header):
            if f.type == np.uint32:
                flipped = "".join(gf.get_flipped_hex(gf.fill_hex_with_zeros(hex(np.uint32(getattr(header, f.name)))[2:], 8), 8))
            elif f.type == List[np.uint32]:
                flipped = ''
                for val in getattr(header, f.name):
                    flipped += "".join(
                        gf.get_flipped_hex(gf.fill_hex_with_zeros(hex(np.uint32(val))[2:], 8), 8))
            else:
                print(f'ERROR {f.type}')
                return
            b.write(binascii.unhexlify(flipped))
        b.write(binascii.unhexlify(file_hex))



def get_image_from_data(header, dimensions, data_hex, save_path):
    format = DXGI_FORMAT[header.TextureFormat]
    print(format)
    img = None
    # if 'RGBA' in format:
    #     try:
    #         img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex))
    #     except ValueError:
    #         return 'Invalid'
    # else:
    bc_decomp(header, data_hex, save_path)
        # return False
    return img

# img = '02B6-0A70'
# pkg = gf.get_pkg_name(img)
# get_image_from_file(f'I:/d2_output_3_0_0_4/{pkg}/{img}.bin', f'imgtests/{img}.dds')
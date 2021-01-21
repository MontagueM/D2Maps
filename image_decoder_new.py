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


old_dir = ''


def get_image_from_file(file_path, all_file_info, save_path=None):
    # pkg_db.start_db_connection()
    file_name = file_path.split('/')[-1].split('.')[0]
    file_pkg = file_path.split('/')[-2]
    # To get the actual image data we need to pull this specific file's data from the database as it references its file
    # in its RefID.
    # TODO FIX THIS REMOVE IT BAD BAD BAD
    # entries = pkg_db.get_entries_from_table(file_pkg, 'FileName, RefID, RefPKG, FileType')
    # this_entry = [x for x in entries if x[0] == file_name][0]
    entry = all_file_info[file_name]
    ref_file_name = f'{entry["RefPKG"][2:]}-{gf.fill_hex_with_zeros(entry["RefID"][2:], 4)}'
    ref_pkg = gf.get_pkg_name(ref_file_name)
    if entry['FileType'] == 'Texture Header':
        header_hex = gf.get_hex_data(file_path)
        data_hex = gf.get_hex_data(f'I:/d2_output_3_0_2_0/{ref_pkg}/{ref_file_name}.bin')
    elif entry['FileType'] == 'Texture Data':
        print('Only pass through header please, cba to fix this.')
        return
    else:
        print(f"File given is not texture data or header of type {entry['FileType']}")
        return
    header = get_header(header_hex)
    dimensions = [header.Width, header.Height]
    large_tex_hash = gf.get_flipped_hex(hex(header.LargeTextureHash)[2:].upper(), 8)
    # print(large_tex_hash)
    if large_tex_hash != 'FFFFFFFF':
        large_file = gf.get_file_from_hash(large_tex_hash)
        pkg_name = gf.get_pkg_name(large_file)
        data_hex = gf.get_hex_data(f'I:/d2_output_3_0_2_0/{pkg_name}/{large_file}.bin')
    print(file_name, ref_file_name)
    img = get_image_from_data(header, dimensions, data_hex, save_path)
    if img:
        if save_path:
            img.save(f'{save_path}')
        else:
            img.save(f'I:/d2_output_3_0_2_0_images/{file_pkg}/{file_name}.png')
            img.show()


def bc_decomp(header, data_hex, save_path):
    """
    8 bytes per 4x4 pixel block
    dwFourCC "DXT1"
    dwFlags DDS_FOURCC

    the compression-related bits are:
    - dwFlags having 0x8 instead of 0x80000
    - dwPitchOrLinearSize might change
    - dwPFFlags 0x40 instead of 0x4
    """
    def conv(num):
        return "".join(gf.get_flipped_hex(gf.fill_hex_with_zeros(hex(np.uint32(num))[2:], 8), 8))

    form = DXGI_FORMAT[header.TextureFormat]
    # if DXT1 BC1 or BC4 block_size = 8 else block_size = 16
    if 'BC1' in form or 'BC4' in form:
        block_size = 8
    else:
        block_size = 16

    # block-compressed
    if 'BC' in form:
        pitch_or_linear_size = max(1, (header.Width+3)/4)*block_size
    elif 'R8GB_B8G8' in form or 'G8R8_G8B8' in form:
        # R8G8_B8G8 or G8R8_G8B8
        pitch_or_linear_size =((header.Width + 1) >> 1) * 4
    else:
        # Any other
        print('other')
        bits_per_pixel = 8  # Assuming its always 8 here which it should be as non-HDR
        # pitch_or_linear_size = (header.Width * bits_per_pixel + 8) / 8
        pitch_or_linear_size = ((header.Width + 1) >> 1) * 4
        # pitch_or_linear_size = header.Width * 32
        # pitch_or_linear_size = 0


    # Compressed
    text_header1 = f'444453207C00000007100800{conv(header.Height)}{conv(header.Width)}{pitch_or_linear_size}000000000000000000000000000000000000000000000000000' \
                  '0000000000000000000000000000000000000000000000000000020000000050000004458313000000000000000000000000' \
                  f'000000000000000000010000000000000000000000000000000000000{conv(header.TextureFormat)}0300000000000000{conv(header.TA)}01000000'

    # Uncompressed
    text_header2 = f'444453207C0000000E100000{conv(header.Height)}{conv(header.Width)}{pitch_or_linear_size}000000000000000000000000000000000000000000000000000' \
                  '000000000000000000000000000000000000000000000000000020000000440000004458313032000000000000000000000' \
                  f'00000000000000000001000000000000000000000000000000000000{header.TextureFormat}03000000000000000100000001000000'
    bc1_header = DDSHeader()  # 0x0
    bc1_header.MagicNumber = int('20534444', 16)  # 0x4
    bc1_header.dwSize = 124  # 0x8
    bc1_header.dwFlags = (0x1 + 0x2 + 0x4 + 0x1000) + 0x8
    bc1_header.dwHeight = header.Height  # 0xC
    bc1_header.dwWidth = header.Width  # 0x10
    bc1_header.dwPitchOrLinearSize = pitch_or_linear_size  # 0x14
    bc1_header.dwDepth = 0
    bc1_header.dwMipMapCount = 0
    bc1_header.dwReserved1 = [0]*11
    bc1_header.dwPFSize = 32
    bc1_header.dwPFRGBBitCount = 0
    bc1_header.dwPFRGBBitCount = 32
    bc1_header.dwPFRBitMask = 0xFF  # RGBA so FF first, but this is endian flipped
    bc1_header.dwPFGBitMask = 0xFF00
    bc1_header.dwPFBBitMask = 0xFF0000
    bc1_header.dwPFABitMask = 0xFF000000

    ''' Uncompressed
    bc1_header.dwPFFourCC = 0
    bc1_header.dwPFRGBBitCount = 32
    bc1_header.dwPFRBitMask = 0xFF  # RGBA so FF first, but this is endian flipped
    bc1_header.dwPFGBitMask = 0xFF00
    bc1_header.dwPFBBitMask = 0xFF0000
    bc1_header.dwPFABitMask = 0xFF000000
    '''

    # bc1_header.dwCaps = 0x1000 + 0x8
    # bc1_header.dwCaps2 = 0x200 + 0x400 + 0x800 + 0x1000 + 0x2000 + 0x4000 + 0x8000  # All faces for cubemap
    bc1_header.dwCaps = 0x1000
    bc1_header.dwCaps2 = 0
    bc1_header.dwCaps3 = 0
    bc1_header.dwCaps4 = 0
    bc1_header.dwReserved2 = 0
    if 'BC' in form:
        bc1_header.dwPFFlags = 0x1 + 0x4  # contains alpha data + contains uncompressed RGB data
        bc1_header.dwPFFourCC = int.from_bytes(b'\x44\x58\x31\x30', byteorder='little')
        bc1_header.dxgiFormat = header.TextureFormat
        bc1_header.resourceDimension = 3  # DDS_DIMENSION_TEXTURE2D
        if header.TA % 6 == 0:
            bc1_header.miscFlag = 4
            bc1_header.arraySize = int(header.TA / 6)
        else:
            bc1_header.miscFlag = 0
            bc1_header.arraySize = 1
            # return  # Used to only export cubemaps
        # int(((int(bc1_header.dwWidth) * int(bc1_header.dwHeight)) + 320) / c_data_size)
    else:
        bc1_header.dwPFFlags = 0x1 + 0x40  # contains alpha data + contains uncompressed RGB data
        bc1_header.dwPFFourCC = 0
        bc1_header.miscFlag = 0
        bc1_header.arraySize = 1
        bc1_header.miscFlags2 = 0x1 #?
        # return
    # print(f'Array size {bc1_header.arraySize}')
    # if 'BC' in DXGI_FORMAT[header.TextureFormat]:
    #     with open(save_path, 'wb') as b:
    #         b.write(binascii.unhexlify(text_header1))
    #         b.write(binascii.unhexlify(data_hex))
    # else:
    write_file(bc1_header, header, data_hex, save_path)



def write_file(header, ogheader, file_hex, save_path):
    # DDS currently broken
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
    # img = Image.frombytes('RGBA', [ogheader.Width, ogheader.Height], bytes.fromhex(file_hex))
    # img.save(save_path[:-4] + '.png')


def get_image_from_data(header, dimensions, data_hex, save_path):
    format = DXGI_FORMAT[header.TextureFormat]
    print(format)
    img = None
    if 'R8G8B8A8' in format:
        bc_decomp(header, data_hex, save_path)
        try:
            return
            img = Image.frombytes('RGBA', dimensions, bytes.fromhex(data_hex))
            img.save(save_path[:-4] + '.png')
        except ValueError:
            return 'Invalid'
    else:
        bc_decomp(header, data_hex, save_path)
        # return False
    # return img

if __name__ == '__main__':
    img = '0172-06AC'
    pkg = gf.get_pkg_name(img)
    pkg_db.start_db_connection()
    all_file_info = {x: y for x,y in {x[0]: dict(zip(['RefID', 'RefPKG', 'FileType'], x[1:])) for x in
                    pkg_db.get_entries_from_table('Everything', 'FileName, RefID, RefPKG, FileType')}.items() if y['FileType'] == 'Texture Header'}
    get_image_from_file(f'I:/d2_output_3_0_2_0/{pkg}/{img}.bin', all_file_info, f'imgtests/edz_0219/{img}.dds')
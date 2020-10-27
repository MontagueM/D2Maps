import model_extractor as met
import gf
import os
import time
import shutil
import struct


def get_shader(model_file: met.ModelFile, submesh: met.Submesh, all_file_info):
    shader = met.File(uid=submesh.material.fhex[0x2C8 * 2:0x2C8 * 2 + 8])
    shader.get_file_from_uid()
    shader_ref = f"{all_file_info[shader.name]['RefPKG'][2:]}-{all_file_info[shader.name]['RefID'][2:]}"
    get_decompiled_hlsl(shader_ref, model_file.uid)
    convert_hlsl(submesh.material, submesh.textures, shader_ref, model_file.uid, all_file_info)


def get_decompiled_hlsl(shader_ref, uid):
    gf.mkdir(f'C:/d2_model_temp/texture_models/{uid}/hlsl/')
    pkg_name = gf.get_pkg_name(shader_ref)
    os.system(f'start 3dmigoto_cmddecompiler/decomp.exe -D C:/d2_output/{pkg_name}/{shader_ref}.bin')
    time.sleep(1)
    shutil.move(f'C:/d2_output/{pkg_name}/{shader_ref}.hlsl', f'C:/d2_model_temp/texture_models/{uid}/hlsl/{shader_ref}.hlsl')
    print(f'Decompiled and moved shader {shader_ref}.hlsl')


def convert_hlsl(material, textures, shader_ref, uid, all_file_info):
    lines_to_write = []
    cbuffer1 = get_cbuffer_from_file(material, all_file_info)
    with open(f'C:/d2_model_temp/texture_models/{uid}/hlsl/{shader_ref}.hlsl', 'r') as h:
        lines_to_write.append('#pragma once')

    with open(f'C:/d2_model_temp/texture_models/{uid}/hlsl/{shader_ref}.usf', 'w') as u:
        u.write(lines_to_write)


def get_cbuffer_from_file(material, all_file_info):
    cbuffer_out = []
    cbuffer_header = met.File(uid=material.fhex[0x34C*2:0x34C*2+8])
    cbuffer_header.get_file_from_uid()
    cbuffer_ref = met.File(name=f"{all_file_info[cbuffer_header.name]['RefPKG'][2:]}-{all_file_info[cbuffer_header.name]['RefID'][2:]}")
    cbuffer_ref.get_hex_data()
    cbuffer = [struct.unpack('f', bytes.fromhex(cbuffer_ref.fhex[i:i + 8]))[0] for i in range(0, len(cbuffer_ref.fhex), 8)]
    for i in range(0, len(cbuffer), 4):
        cbuffer_out.append(f'   float4({cbuffer[i]}, {cbuffer[i+1]}, {cbuffer[i+2]}, {cbuffer[i+3]}),')
    return cbuffer_out

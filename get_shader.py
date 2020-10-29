import model_extractor as met
import gf
import os
import time
import shutil
import struct
import re


def get_shader(model_file, submesh, all_file_info, name):
    shader = met.File(uid=submesh.material.fhex[0x2C8 * 2:0x2C8 * 2 + 8])
    shader.get_file_from_uid()
    shader_ref = f"{all_file_info[shader.name]['RefPKG'][2:]}-{all_file_info[shader.name]['RefID'][2:]}"
    get_decompiled_hlsl(shader_ref, model_file.uid)
    convert_hlsl(submesh.material, submesh.textures, shader_ref, model_file.uid, all_file_info, name)


def get_shader_from_mat(material, textures, cbuffer_offsets, all_file_info, custom_dir):
    # if material.name in os.listdir(custom_dir):
    #     return
    shader = met.File(uid=material.fhex[0x2C8 * 2:0x2C8 * 2 + 8])
    shader.get_file_from_uid()
    shader_ref = f"{all_file_info[shader.name]['RefPKG'][2:]}-{all_file_info[shader.name]['RefID'][2:]}"
    get_decompiled_hlsl(shader_ref, custom_dir)
    convert_hlsl(material, textures, cbuffer_offsets, shader_ref, custom_dir, all_file_info)


def get_decompiled_hlsl(shader_ref, custom_dir):
    gf.mkdir(custom_dir)
    pkg_name = gf.get_pkg_name(shader_ref)
    os.system(f'start hlsl/decomp.exe -D C:/d2_output/{pkg_name}/{shader_ref}.bin')
    time.sleep(1)
    shutil.move(f'C:/d2_output/{pkg_name}/{shader_ref}.hlsl', f'{custom_dir}/{shader_ref}.hlsl')
    print(f'Decompiled and moved shader {shader_ref}.hlsl')


def convert_hlsl(material, textures, cbuffer_offsets, shader_ref, custom_dir, all_file_info, name=None):
    print(f'Material {material.name}')
    lines_to_write = []

    # Getting info from material
    cbuffers = get_all_cbuffers_from_file(material, cbuffer_offsets, all_file_info)


    # Getting info from hlsl file
    with open(f'{custom_dir}/{shader_ref}.hlsl', 'r') as h:
        text = h.readlines()
        instructions = get_instructions(text)
        cbuffer_text = get_cbuffer_text(cbuffers, text)
        inputs, outputs = get_in_out(text, instructions)
        input_append1, input_append2 = get_inputs_append(inputs)
        texs = get_texs(text)
        params, params_end = get_params(texs)
        tex_comments = get_tex_comments(textures)
        lines_to_write.append('#pragma once\n')
        lines_to_write.append(tex_comments)
        lines_to_write.append(cbuffer_text)
        # lines_to_write.append(f'static float4 cb0[{cbuffer_length}] = \n' + '{\n' + f'{cbuffer1}\n' + '};\n')
        lines_to_write.append(input_append1)
        lines_to_write.append('\n\nstruct shader {\nfloat4 main(\n')
        lines_to_write.append(params)
        lines_to_write.append(input_append2)
        lines_to_write.append(f'    float4 {",".join(outputs)};\n')
        lines_to_write.append(instructions)
        lines_to_write.append('}\n};\n\nshader s;\n\n' + f'return s.main({params_end}, tx);')

    # Change to 3 for all outputs, currently just want base colour
    for i in range(1):
        if name:
            open_dir = f'{custom_dir}/{name}_{shader_ref}_o{i}.usf'
        else:
            open_dir = f'{custom_dir}/{material.name}_o{i}.usf'
        with open(open_dir, 'w') as u:
        # with open(f'hlsl/.usf', 'w') as u:
            # TODO convert to an array write, faster
            for line in lines_to_write:
                if 'return' in line:
                    line = line.replace('return;', f'return o{i};')
                u.write(line)
            print(f'Wrote to usf {open_dir}')
        print('')


def get_cbuffer_text(cbuffers, text):
    ret = ''
    # This all assumes there won't be two cbuffers of the same length
    cbuffer_to_write = {}
    text_cbuffers = {}
    read = False
    for line in text:
        if 'cbuffer' in line:
            read = True
        if read:
            if 'register' in line:
                name = line.split(' ')[1]
            elif 'float4' in line:
                size = int(line.split('[')[1].split(']')[0])
            elif '}' in line:
                text_cbuffers[size] = name
                read = False
    for length, data in cbuffers.items():
        if length in text_cbuffers.keys():
            name = text_cbuffers[length]
            cbuffer_to_write[name] = [data, length]

    # As we don't know where to find cb12 yet
    if 'cb12' in text_cbuffers.values():
        cbuffer_to_write['cb12'] = ['float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1),', 8]

    for name, packed in cbuffer_to_write.items():
        data = packed[0]
        length = packed[1]
        ret += f'static float4 {name}[{length}] = \n' + '{\n' + f'{data}\n' + '};\n'

    return ret


def get_tex_comments(textures):
    comments = ''
    comments += f'//{textures}\n'
    for i, t in enumerate(textures):
        comments += f'// t{i} is {t}\n'
    return comments


def get_inputs_append(inputs):
    input_append1 = ''
    input_append2 = ''
    for inp in inputs:
        inps = inp.split(' ')
        if 'TEXCOORD' in inp:
            if 'float4' in inp:
                write = f'\nstatic {inps[2]} {inps[3]} = ' + '{1, 1, 1, 1};\n'
            elif 'float3' in inp:
                write = f'\nstatic {inps[2]} {inps[3]} = ' + '{1, 1, 1};\n'
            input_append2 += f'    {inps[3]}.xy = {inps[3]}.xy * tx;\n'
        elif 'SV_isFrontFace0' in inp:
            write = f'\nstatic {inps[2]} {inps[3]} = 1;\n'
        else:
            raise Exception('Input not recognised.')
        input_append1 += write
    return input_append1, input_append2


def get_params(texs):
    params = ''
    params_end = ''
    texs = texs[::-1]
    for t in texs:
        if texs[-1] == t:
            params += f'  float4 {t},\n   float2 tx)\n' + '{\n'
            params_end += t
        else:
            params += f'  float4 {t},\n'
            params_end += f'{t}, '
    print('')
    return params, params_end


def get_texs(text):
    texs = []
    for line in text:
        if 'Texture2D<float4>' in line:
            texs.append(line.split(' ')[1])
    return texs


def get_instructions(text):
    instructions = []
    care = False
    read = False
    for line in text:
        if read:
            if 'Sample' in line:
                equal = line.split('=')[0]
                to_sample = [x for x in line.split(' ') if x != ''][2].split('.')[0]
                samplestate = int(line.split('(')[1][1])
                uv = line.split(', ')[1].split(')')[0]
                dot_after = line.split(').')[1]
                line = f'{equal}= Material_Texture2D_{to_sample[1:]}.SampleLevel(Material_Texture2D_{samplestate-1}Sampler, {uv}, 0).{dot_after}'
            instructions.append('  ' + line)
            if 'return;' in line:
                ret = ''.join(instructions)
                # cmp seems broken
                ret = ret.replace('cmp', '')
                # discard doesnt work in ue4 hlsl
                ret = ret.replace('discard', '{ o0.w = 0; }')
                return ret
        elif 'void main(' in line:
            care = True
        elif care and '{' in line:
            read = True


def get_in_out(text, instructions):
    inputs = []
    outputs = []
    read = False
    for line in text:
        if 'void main(' in line:
            read = True
            continue
        if read:
            if 'out' in line:
                outputs.append(line.split(' ')[4])
            elif '{' in line:
                return inputs, outputs
            else:
                inp = line.split(' ')[3]
                if inp in instructions:
                    inputs.append(line[:-1])


def get_all_cbuffers_from_file(material, cbuffer_offsets, all_file_info):
    cbuffers = {}
    # Read cbuffer from file if there
    if material.fhex[0x34C*2:0x34C*2+8] != 'FFFFFFFF':
        cbuffer_header = met.File(uid=material.fhex[0x34C*2:0x34C*2+8])
        cbuffer_header.get_file_from_uid()
        cbuffer_ref = met.File(name=f"{all_file_info[cbuffer_header.name]['RefPKG'][2:]}-{all_file_info[cbuffer_header.name]['RefID'][2:]}")
        cbuffer_ref.get_hex_data()
        data, length = process_cbuffer_data(cbuffer_ref.fhex)
        cbuffers[length] = data

    # Reading from mat file as well in case there's more cbuffers
    # offsets = [m.start() for m in re.finditer('90008080', material.fhex)]
    # If cbuffer is a real cbuffer we'll read it and output it
    for offset in cbuffer_offsets:
        offset += 16
        count = int(gf.get_flipped_hex(material.fhex[offset-16:offset-8], 8), 16)
        if count != 1:
            data, length = process_cbuffer_data(material.fhex[offset+16:offset+16+32*count])
            cbuffers[length] = data
        # else:
            # raise Exception('No cbuffer found.')
    return cbuffers


def process_cbuffer_data(fhex):
    cbuffer_out = []
    cbuffer = [struct.unpack('f', bytes.fromhex(fhex[i:i + 8]))[0] for i in
               range(0, len(fhex), 8)]
    for i in range(0, len(cbuffer), 4):
        cbuffer_out.append(f'   float4({cbuffer[i]}, {cbuffer[i + 1]}, {cbuffer[i + 2]}, {cbuffer[i + 3]}),')
    return '\n'.join(cbuffer_out), int(len(cbuffer) / 4)

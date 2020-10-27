import model_extractor as met
import gf
import os
import time
import shutil
import struct


def get_shader(model_file, submesh, all_file_info):
    shader = met.File(uid=submesh.material.fhex[0x2C8 * 2:0x2C8 * 2 + 8])
    shader.get_file_from_uid()
    shader_ref = f"{all_file_info[shader.name]['RefPKG'][2:]}-{all_file_info[shader.name]['RefID'][2:]}"
    get_decompiled_hlsl(shader_ref, model_file.uid)
    convert_hlsl(submesh.material, submesh.textures, shader_ref, model_file.uid, all_file_info)


def get_decompiled_hlsl(shader_ref, uid):
    gf.mkdir(f'C:/d2_model_temp/texture_models/{uid}/hlsl/')
    pkg_name = gf.get_pkg_name(shader_ref)
    os.system(f'start hlsl/decomp.exe -D C:/d2_output/{pkg_name}/{shader_ref}.bin')
    time.sleep(1)
    shutil.move(f'C:/d2_output/{pkg_name}/{shader_ref}.hlsl', f'C:/d2_model_temp/texture_models/{uid}/hlsl/{shader_ref}.hlsl')
    print(f'Decompiled and moved shader {shader_ref}.hlsl')


def convert_hlsl(material, textures, shader_ref, uid, all_file_info):
    print(f'Material {material.name}')
    lines_to_write = []

    # Getting info from material
    cbuffer1, cbuffer_length = get_cbuffer_from_file(material, all_file_info)

    # Getting info from hlsl file
    with open(f'C:/d2_model_temp/texture_models/{uid}/hlsl/{shader_ref}.hlsl', 'r') as h:
        text = h.readlines()
        instructions = get_instructions(text)
        inputs, outputs = get_in_out(text, instructions)
        input_append = get_inputs_append(inputs)
        texs = get_texs(text)
        params, params_end = get_params(texs)
        lines_to_write.append('#pragma once\n')
        lines_to_write.append(f'static float4 cb0[{cbuffer_length}] = \n{cbuffer1}\n' + '};\n')
        lines_to_write.append(input_append)
        lines_to_write.append('\n\nstruct shader {\nfloat4 main(\n')
        lines_to_write.append(params)
        lines_to_write.append(f'    float4 {",".join(outputs)};\n')
        lines_to_write.append(instructions)
        lines_to_write.append('}\n};\n\nshader s;\n\n' + f'return shader.main({params_end});')

    for i in range(3):
        with open(f'C:/d2_model_temp/texture_models/{uid}/hlsl/{shader_ref}_o{i}.usf', 'w') as u:
        # with open(f'hlsl/.usf', 'w') as u:
            # TODO convert to an array write, faster
            for line in lines_to_write:
                if 'return' in line:
                    line = line.replace('return;', f'return o{i};')
                u.write(line)
            print('Wrote to file')
        print('')


def get_inputs_append(inputs):
    input_append = ''
    for inp in inputs:
        inps = inp.split(' ')
        if 'TEXCOORD' in inp:
            write = f'\nstatic {inps[2]} {inps[3]} = ' + '{1, 1, 1, 1};\n'
        elif 'SV_isFrontFace0' in inp:
            write = f'\nstatic {inps[2]} {inps[3]} = 1;\n'
        else:
            raise Exception('Input not recognised.')
        input_append += write
    return input_append


def get_params(texs):
    params = ''
    params_end = ''
    texs = texs[::-1]
    for t in texs:
        if texs[-1] == t:
            params += f'  float4 {t})\n' + '{\n'
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
                to_sample = line.split(' ')[4].split('.')[0]
                samplestate = int(line.split('(')[1][1])
                uv = line.split(', ')[1].split(')')[0]
                dot_after = line.split(').')[1]
                line = f'{equal}= Material_Texture2D_{to_sample[1:]}.SampleLevel(Material_Texture2D_{samplestate-1}Sampler, {uv}, 0).{dot_after}'
            instructions.append('  ' + line)
            if 'return;' in line:
                ret = ''.join(instructions)
                # cmp seems broken
                return ret.replace('cmp', '')
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


def get_cbuffer_from_file(material, all_file_info):
    if material.fhex[0x34C*2:0x34C*2+8] == 'FFFFFFFF':
        # No cbuffer file so trying to see if theres a cbuffer within the file instead
        offset_9000 = material.fhex.find('90008080')
        if int(gf.get_flipped_hex(material.fhex[offset_9000-16:offset_9000-8], 8), 16) != 1:
            if 'BD9F8080' not in material.fhex[offset_9000:]:
                # Assuming that it's always at the end of the file
                return process_cbuffer_data(material.fhex[offset_9000+16:])
            else:
                raise Exception('Cbuffer not end of file')
        else:
            raise Exception('No cbuffer found.')
    else:
        cbuffer_header = met.File(uid=material.fhex[0x34C*2:0x34C*2+8])
        cbuffer_header.get_file_from_uid()
        cbuffer_ref = met.File(name=f"{all_file_info[cbuffer_header.name]['RefPKG'][2:]}-{all_file_info[cbuffer_header.name]['RefID'][2:]}")
        cbuffer_ref.get_hex_data()
        return process_cbuffer_data(cbuffer_ref.fhex)


def process_cbuffer_data(fhex):
    cbuffer_out = []
    cbuffer = [struct.unpack('f', bytes.fromhex(fhex[i:i + 8]))[0] for i in
               range(0, len(fhex), 8)]
    for i in range(0, len(cbuffer), 4):
        cbuffer_out.append(f'   float4({cbuffer[i]}, {cbuffer[i + 1]}, {cbuffer[i + 2]}, {cbuffer[i + 3]}),')
    return '\n'.join(cbuffer_out), int(len(cbuffer) / 4)

from dataclasses import dataclass, fields
import numpy as np
import struct
import model_unpacker_textures as mut
import scipy.spatial
import pkg_db
import fbx
import pyfbx as pfb
import gf


@dataclass
class CountEntry:
    Field0: np.uint16 = np.uint32(0)
    Field2: np.uint32 = np.uint32(0)
    Field6: np.uint16 = np.uint32(0)


def get_hex_data(direc):
    t = open(direc, 'rb')
    h = t.read().hex().upper()
    return h


def get_flipped_hex(h, length):
    if length % 2 != 0:
        print("Flipped hex length is not even.")
        return None
    return "".join(reversed([h[:length][i:i + 2] for i in range(0, length, 2)]))


def get_header(file_hex, header):
    # The header data is 0x16F bytes long, so we need to x2 as python reads each nibble not each byte

    for f in fields(header):
        if f.type == np.uint32:
            flipped = "".join(get_flipped_hex(file_hex, 8))
            value = np.uint32(int(flipped, 16))
            setattr(header, f.name, value)
            file_hex = file_hex[8:]
        elif f.type == np.uint16:
            flipped = "".join(get_flipped_hex(file_hex, 4))
            value = np.uint16(int(flipped, 16))
            setattr(header, f.name, value)
            file_hex = file_hex[4:]
    return header


def unpack_map(main_file, all_file_info, ginsor_debug, scale_100x, folder_name='Other'):
    # If the file is too large you can uncomment the LARGE stuff

    fbx_map = pfb.Model()
    fbx_map.create_node()

    scale_hex, transform_hex, model_refs_hex, copy_count_hex = get_hex_from_pkg(main_file)

    rotations, locations, map_scaler = get_transform_data(transform_hex, scale_hex)
    model_refs = get_model_refs(model_refs_hex)
    print(len(model_refs), len(rotations))
    copy_counts = get_copy_counts(copy_count_hex)
    transforms_array = get_transforms_array(model_refs, copy_counts, rotations, locations, map_scaler)
    # if main_file == '0932-000001FE':  # LARGE
    #     LARGE_total_end_files = 2  # LARGE
    #     for i in range(LARGE_total_end_files):  # LARGE
    #         split_len = int(len(transforms_array)/LARGE_total_end_files)  # LARGE
    #         obj_strings = get_model_obj_strings(transforms_array[split_len*i:split_len*(i+1)], version, scale_coords_extra, modifiers)  # LARGE
    #         write_obj_strings(obj_strings, folder_name, main_file, i)  # LARGE
    # else:
    #     return  # LARGE

    fbx_map = compute_coords(transforms_array, all_file_info, fbx_map, ginsor_debug, scale_100x)
    write_fbx(fbx_map, folder_name, main_file)


def get_hex_from_pkg(file):
    pkgs_dir = 'C:/d2_output/'

    main_pkg = gf.get_pkg_name(file)
    main_hex = get_hex_data(f'{pkgs_dir}{main_pkg}/{file}.bin')
    scales_file = get_scales_file(main_hex)
    scales_pkg = gf.get_pkg_name(scales_file)
    scale_hex = get_hex_data(f'{pkgs_dir}{scales_pkg}/{scales_file}.bin')[48 * 2:]

    transform_count = int(get_flipped_hex(main_hex[64*2:64*2+4], 4), 16)
    transform_offset = 192
    transform_length = transform_count*48
    transform_hex = main_hex[transform_offset*2:transform_offset*2 + transform_length*2]

    entry_count = int(get_flipped_hex(main_hex[88*2:88*2+4], 4), 16)
    model_offset = transform_offset + transform_length + 32
    model_length = entry_count * 4
    model_refs_hex = main_hex[model_offset*2:model_offset*2 + model_length*2]

    copy_offset = model_offset + model_length + int(main_hex[model_offset*2+model_length*2:].find('90718080')/2) + 8
    copy_count_hex = main_hex[copy_offset*2:]

    return scale_hex, transform_hex, model_refs_hex, copy_count_hex


def get_scales_file(main_hex):
    file_hash = main_hex[24*2:24*2+8]
    file_name = gf.get_file_from_hash(file_hash)
    return file_name


def get_transform_data(transform_hex, scale_hex):
    rotation_entries_hex = [transform_hex[i:i + 48 * 2] for i in range(0, len(transform_hex), 48 * 2)]

    rotations = []
    for e in rotation_entries_hex:
        h = e[:16 * 2]
        hex_floats = [h[i:i + 8] for i in range(0, len(h), 8)]
        floats = []
        for hex_float in hex_floats:
            float_value = round(struct.unpack('f', bytes.fromhex(hex_float))[0], 6)
            floats.append(float_value)
        rotations.append(floats)

    map_scaler = []
    for e in rotation_entries_hex:
        float_value = round(struct.unpack('f', bytes.fromhex(e[28*2:32*2]))[0], 6)
        map_scaler.append(float_value)

    locations = []
    for e in rotation_entries_hex:
        loc_hex = e[16 * 2:28 * 2]
        loc_hex_floats = [loc_hex[i:i + 8] for i in range(0, len(loc_hex), 8)]
        location = []
        for hex_float in loc_hex_floats:
            float_value = round(struct.unpack('f', bytes.fromhex(hex_float))[0], 6)
            location.append(float_value)
        locations.append(location)

    return rotations, locations, map_scaler


def get_model_refs(model_refs_hex):
    entries_hex = [model_refs_hex[i:i + 4 * 2] for i in range(0, len(model_refs_hex), 4 * 2)]
    return entries_hex


def get_copy_counts(copy_count_hex):
    entries_hex = [copy_count_hex[i:i + 8 * 2] for i in range(0, len(copy_count_hex), 8 * 2)]
    entries = []
    for e in entries_hex:
        entries.append(get_header(e, CountEntry()))
    return [e.Field0 for e in entries]


def get_transforms_array(model_refs, copy_counts, rotations, location, map_scaler):
    transforms_array = []
    last_index = 0
    for i, model in enumerate(model_refs):
        copies = copy_counts[i]
        transforms = []
        for copy_id in range(copies):
            transforms.append({'rotation': rotations[last_index + copy_id],
                               'location': location[last_index + copy_id],
                               'map_scaler': map_scaler[last_index + copy_id]})
        last_index += copies
        transform_array = [model, transforms]
        transforms_array.append(transform_array)
    return transforms_array


def compute_coords(transforms_array, all_file_info, fbx_map, ginsor_debug, scale_100x):
    max_vert_used = 0
    nums = 0

    for i, transform_array in enumerate(transforms_array):
        model_file = gf.get_file_from_hash(transform_array[0])
        model_data_file = mut.get_model_data_file(model_file)
        submeshes_verts, submeshes_faces = mut.get_verts_faces_data(model_data_file, all_file_info, model_file)
        if not submeshes_verts or not submeshes_faces:
            print('Skipping current model')
            continue
        print(f'Getting obj {i + 1}/{len(transforms_array)} {transform_array[0]} {nums}')

        for copy_id, transform in enumerate(transform_array[1]):
            nums += 1
            # TODO NOTE THERE ARE 6 VERTS PACKED AS IT IS POSX,POSY,POSZ,UV1,UV2,UV3???
            all_index_2_verts = []
            [[[all_index_2_verts.append(z) for z in y] for y in x] for x in submeshes_verts.values()]

            r_verts_data = rotate_verts(all_index_2_verts, transform['rotation'])
            map_scaled_verts = get_map_scaled_verts(r_verts_data, transform['map_scaler'])
            map_moved_verts = get_map_moved_verts(map_scaled_verts, transform['location'], scale_100x)

            offset = 0
            for index_2 in submeshes_verts.keys():
                for index_3 in range(len(submeshes_verts[index_2])):
                    new_verts = map_moved_verts[offset:offset + len(submeshes_verts[index_2][index_3])]
                    offset += len(submeshes_verts[index_2][index_3])
                    adjusted_faces_data, max_vert_used = mut.adjust_faces_data(submeshes_faces[index_2][index_3],
                                                                                          max_vert_used)
                    shifted_faces = shift_faces_down(adjusted_faces_data)
                    fbx_map = add_model_to_fbx_map(fbx_map, shifted_faces, new_verts, f'{transform_array[0]}_{copy_id}_{index_2}_{index_3}')
    return fbx_map


def get_map_scaled_verts(verts_data, map_scaler):
    for i in range(len(verts_data)):
        for j in range(3):
            verts_data[i][j] *= map_scaler
    return verts_data


def get_map_moved_verts(verts_data, location, scale_100x):
    for i in range(len(verts_data)):
        for j in range(3):
            verts_data[i][j] += location[j]
            if scale_100x:
                verts_data[i][j] *= 100
    return verts_data


def shift_faces_down(faces_data):
    a_min = faces_data[0][0]
    for f in faces_data:
        for i in f:
            if i < a_min:
                a_min = i
    for i, f in enumerate(faces_data):
        for j, x in enumerate(f):
            faces_data[i][j] -= a_min - 1
    return faces_data


def add_model_to_fbx_map(fbx_map, faces_data, verts_data, name):
    mesh = fbx.FbxMesh.Create(fbx_map.scene, name)
    controlpoints = [fbx.FbxVector4(x[0], x[1], x[2]) for x in verts_data]
    controlpoint_count = len(controlpoints)
    mesh.InitControlPoints(controlpoint_count)
    for i, p in enumerate(controlpoints):
        mesh.SetControlPointAt(p, i)
    for face in faces_data:
        mesh.BeginPolygon()
        mesh.AddPolygon(face[0]-1)
        mesh.AddPolygon(face[1]-1)
        mesh.AddPolygon(face[2]-1)
        mesh.EndPolygon()

    node = fbx.FbxNode.Create(fbx_map.scene, name)
    node.SetNodeAttribute(mesh)
    fbx_map.scene.GetRootNode().AddChild(node)

    return fbx_map


def rotate_verts(verts_data, rotation_transform, inverse=False):
    r = scipy.spatial.transform.Rotation.from_quat(rotation_transform)
    if len(verts_data) == 3:
        quat_rots = scipy.spatial.transform.Rotation.apply(r, verts_data, inverse=inverse)
    else:
        quat_rots = scipy.spatial.transform.Rotation.apply(r, [[x[0], x[1], x[2]] for x in verts_data], inverse=inverse)
    return quat_rots.tolist()


def write_fbx(fbx_map, folder_name, file_name):
    gf.mkdir(f'C:/d2_maps/{folder_name}_fbx/')
    fbx_map.export(save_path=f'C:/d2_maps/{folder_name}_fbx/{file_name}.fbx', ascii_format=False)
    print('Wrote fbx')


def unpack_folder(pkg_name, ginsor_debug=False, scale_100x=True):
    pkg_db.start_db_connection()
    entries_refid = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, RefID') if y == '0x166D'}
    entries_refpkg = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, RefPKG') if y == '0x0004'}
    entries_size = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, FileSizeB')}
    file_names = sorted(entries_refid.keys(), key=lambda x: entries_size[x])
    all_file_info = {x[0]: dict(zip(['RefID', 'RefPKG', 'FileType'], x[1:])) for x in
                     pkg_db.get_entries_from_table('Everything', 'FileName, RefID, RefPKG, FileType')}
    for file_name in file_names:
        if file_name in entries_refpkg.keys():
            if '1A4A' not in file_name:
                continue
            print(f'Unpacking {file_name}')
            unpack_map(file_name,  all_file_info, ginsor_debug, scale_100x, folder_name=pkg_name)


if __name__ == '__main__':
    unpack_folder('city_tower_d2_0369', ginsor_debug=True, scale_100x=True)

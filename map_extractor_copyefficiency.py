from dataclasses import dataclass, fields
import numpy as np
import struct
import model_extractor as met
import scipy.spatial
import pkg_db
import fbx
import pyfbx as pfb
import gf
import quaternion
import math


@dataclass
class CountEntry:
    Count: np.uint16 = np.uint16(0)
    CumulativeCount: np.uint16 = np.uint16(0)
    ModelRef: np.uint16 = np.uint16(0)
    Unk: np.uint16 = np.uint16(0)


class Map(met.File):
    def __init__(self, name):
        super(Map, self).__init__(name=name)
        self.scales_hex = ''
        self.scales = []
        self.transforms_hex = ''
        self.rotations = []
        self.locations = []
        self.model_refs_hex = ''
        self.model_refs = []
        self.copy_counts_hex = ''
        self.copy_counts = []
        self.fbx_model = None
        self.materials = {}
        self.material_files = []


def get_header(file_hex, header):
    # The header data is 0x16F bytes long, so we need to x2 as python reads each nibble not each byte

    for f in fields(header):
        if f.type == np.uint32:
            flipped = "".join(gf.get_flipped_hex(file_hex, 8))
            value = np.uint32(int(flipped, 16))
            setattr(header, f.name, value)
            file_hex = file_hex[8:]
        elif f.type == np.uint16:
            flipped = "".join(gf.get_flipped_hex(file_hex, 4))
            value = np.uint16(int(flipped, 16))
            setattr(header, f.name, value)
            file_hex = file_hex[4:]
    return header


def unpack_map(main_file, pkg_name, unreal, shaders):
    d2map = Map(name=main_file)
    gf.mkdir(f'I:/maps/{pkg_name}_fbx/')

    d2map.fbx_model = pfb.Model()
    d2map.fbx_model.create_node()
    d2map.fbx_model.scene.GetRootNode().SetRotationActive(True)
    d2map.fbx_model.scene.GetRootNode().SetGeometricRotation(fbx.FbxNode.eSourcePivot, fbx.FbxVector4(53, -77, 165))
    get_hex_from_pkg(d2map)

    get_transform_data(d2map)
    get_model_refs(d2map)
    get_copy_counts(d2map)

    compute_coords(d2map, unreal, shaders)
    if shaders:
        get_shader_info(d2map)
    write_fbx(d2map)


def get_hex_from_pkg(d2map: Map):
    main_hex = gf.get_hex_data(f'I:/d2_output_3_0_0_4/{d2map.get_pkg_name()}/{d2map.name}.bin')
    file_hash = main_hex[24*2:24*2+8]
    scales_file = met.File(name=gf.get_file_from_hash(file_hash))
    d2map.scales_hex = gf.get_hex_data(f'I:/d2_output_3_0_0_4/{scales_file.get_pkg_name()}/{scales_file.name}.bin')[48 * 2:]

    transform_count = int(gf.get_flipped_hex(main_hex[64*2:64*2+4], 4), 16)
    transform_offset = int(main_hex.find('406D8080')/2+8)
    transform_length = transform_count*48
    d2map.transforms_hex = main_hex[transform_offset*2:transform_offset*2 + transform_length*2]

    entry_count = int(gf.get_flipped_hex(main_hex[88*2:88*2+4], 4), 16)
    model_offset = transform_offset + transform_length + 32
    model_length = entry_count * 4
    d2map.model_refs_hex = main_hex[model_offset*2:model_offset*2 + model_length*2]

    copy_offset = model_offset + model_length + int(main_hex[model_offset*2+model_length*2:].find('286D8080')/2) + 8
    d2map.copy_counts_hex = main_hex[copy_offset*2:]


def get_transform_data(d2map: Map):
    rotation_entries_hex = [d2map.transforms_hex[i:i + 48 * 2] for i in range(0, len(d2map.transforms_hex), 48 * 2)]

    for e in rotation_entries_hex:
        h = e[:16 * 2]
        hex_floats = [h[i:i + 8] for i in range(0, len(h), 8)]
        floats = []
        for hex_float in hex_floats:
            float_value = round(struct.unpack('f', bytes.fromhex(hex_float))[0], 6)
            floats.append(float_value)
        d2map.rotations.append(floats)

        float_value = round(struct.unpack('f', bytes.fromhex(e[28*2:32*2]))[0], 6)
        d2map.scales.append(float_value)

        loc_hex = e[16 * 2:28 * 2]
        loc_hex_floats = [loc_hex[i:i + 8] for i in range(0, len(loc_hex), 8)]
        location = []
        for hex_float in loc_hex_floats:
            float_value = round(struct.unpack('f', bytes.fromhex(hex_float))[0], 6)
            location.append(float_value)
        d2map.locations.append(location)


def get_model_refs(d2map: Map):
    d2map.model_refs = [d2map.model_refs_hex[i:i + 4 * 2] for i in range(0, len(d2map.model_refs_hex), 4 * 2)]


def get_copy_counts(d2map: Map):
    entries_hex = [d2map.copy_counts_hex[i:i + 8 * 2] for i in range(0, len(d2map.copy_counts_hex), 8 * 2)]
    entries = []
    for e in entries_hex:
        entries.append(get_header(e, CountEntry()))
    for entry in entries:
        d2map.copy_counts.append({'UID': d2map.model_refs[entry.ModelRef], 'Count': entry.Count})
    print('')


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


def compute_coords(d2map: Map, unreal, shaders):
    nums = 0

    d2map.fbx_model.scene.GetRootNode().SetGeometricRotation(fbx.FbxNode.eSourcePivot, fbx.FbxVector4(-90, -25, -80))
    for i, data in enumerate(d2map.copy_counts):
        model_ref = data['UID']
        copy_count = data['Count']
        print(f'Getting obj {i + 1}/{len(d2map.copy_counts)} {model_ref} {nums}')
        if model_ref == '3819C680':
            a = 0
        model_file = met.ModelFile(uid=model_ref)
        model_file.get_model_data_file()
        ret = met.get_model_data(model_file, all_file_info)
        if not ret:
            nums += copy_count
            continue
        met.get_submeshes(model_file)
        # met.get_materials(model_file)

        """Could make more efficient by just duping models instead of remaking them here."""
        max_vert_used = 0
        for j, model in enumerate(model_file.models):
            for k, submesh in enumerate(model.submeshes):
                if submesh.type == 769 or submesh.type == 770 or submesh.type == 778 or submesh.type == 'New':
                    for cc in range(copy_count):
                        name = f'{model_ref}_{cc}_{j}_{k}'
                        if cc == 0:
                            submesh.faces, max_vert_used = met.adjust_faces_data(submesh.faces, max_vert_used)
                            submesh.faces = met.shift_faces_down(submesh.faces)
                            mesh = create_mesh(d2map, submesh, name)
                        node = fbx.FbxNode.Create(d2map.fbx_model.scene, name)
                        node.SetRotationActive(True)
                        rot = d2map.rotations[nums+cc]
                        node.SetNodeAttribute(mesh)
                        loc_rot = d2map.locations[nums+cc]
                        r = scipy.spatial.transform.Rotation.from_quat(rot).as_euler('xyz', degrees=True)
                        node.SetGeometricRotation(fbx.FbxNode.eSourcePivot, fbx.FbxVector4(-r[0]-90, r[1]-180, r[2]))
                        node.SetGeometricTranslation(fbx.FbxNode.eSourcePivot, fbx.FbxVector4(loc_rot[0]*100, loc_rot[1]*100, loc_rot[2]*100))
                        node.SetGeometricScaling(fbx.FbxNode.eSourcePivot, fbx.FbxVector4(d2map.scales[nums+cc]*100, d2map.scales[nums+cc]*100, d2map.scales[nums+cc]*100))

                        if unreal:
                            d2map.fbx_model.scene.GetRootNode().AddChild(node)
                        else:
                            tempnode = fbx.FbxNode.Create(d2map.fbx_model.scene, name + 'k')
                            tempnode.AddChild(node)
                            tempnode.SetRotationActive(True)
                            tempnode.SetGeometricRotation(fbx.FbxNode.eSourcePivot,
                                                      fbx.FbxVector4(-90, 180, 0))
                            d2map.fbx_model.scene.GetRootNode().AddChild(tempnode)
        nums += copy_count


def get_map_scaled_verts(submesh: met.Submesh, map_scaler):
    for i in range(len(submesh.adjusted_pos_verts)):
        for j in range(3):
            submesh.adjusted_pos_verts[i][j] *= map_scaler


def get_map_moved_verts(submesh: met.Submesh, location, scale):
    for i in range(len(submesh.adjusted_pos_verts)):
        for j in range(3):
            submesh.adjusted_pos_verts[i][j] += location[j]# * scale
            # submesh.adjusted_pos_verts[i][j] *= 100


def rotate_verts(verts, rotation, inverse=False):
    r = scipy.spatial.transform.Rotation.from_quat(rotation)
    if len(verts) == 3:
        quat_rots = scipy.spatial.transform.Rotation.apply(r, verts, inverse=inverse)
    else:
        quat_rots = scipy.spatial.transform.Rotation.apply(r, [[x[0], x[1], x[2]] for x in verts], inverse=inverse)
    return quat_rots.tolist()


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


def get_shader_info(d2map: Map):
    for material in d2map.material_files:
        if material.name == '0222-0CAF':
            print('')
        cbuffer_offsets, texture_offset = met.get_mat_tables(material)
        textures = met.get_material_textures(material, texture_offset, custom_dir=f'I:/maps/{d2map.pkg_name}_fbx/textures/')
        met.get_shader_file(material, textures, cbuffer_offsets, all_file_info, custom_dir=f'I:/maps/{d2map.pkg_name}_fbx/shaders/')


def add_model_to_fbx_map(d2map: Map, model_file: met.ModelFile, submesh: met.Submesh, name, shaders):
    node, mesh = create_mesh(d2map, submesh, name)
    # met.get_submesh_textures(model_file, submesh, custom_dir=f'C:/d2_maps/{d2map.pkg_name}_fbx/textures/')
    if not mesh.GetLayer(0):
        mesh.CreateLayer()
    layer = mesh.GetLayer(0)
    if shaders:
        apply_shader(d2map, submesh, node)
    # apply_diffuse(d2map, submesh, node)
    create_uv(mesh, name, submesh, layer)
    node.SetShadingMode(fbx.FbxNode.eTextureShading)
    d2map.fbx_model.scene.GetRootNode().AddChild(node)


def apply_shader(d2map: Map, submesh: met.Submesh, node):
    lMaterialName = f'{submesh.material.name}'
    if lMaterialName in d2map.materials.keys():
        node.AddMaterial(d2map.materials[lMaterialName])
        return
    lMaterial = fbx.FbxSurfacePhong.Create(d2map.fbx_model.scene, lMaterialName)
    d2map.materials[lMaterialName] = lMaterial
    d2map.material_files.append(met.File(name=lMaterialName))
    # lMaterial.DiffuseFactor.Set(1)
    lMaterial.ShadingModel.Set('Phong')
    node.AddMaterial(lMaterial)


def create_mesh(d2map: Map, submesh: met.Submesh, name):
    mesh = fbx.FbxMesh.Create(d2map.fbx_model.scene, name)
    controlpoints = [fbx.FbxVector4(-x[0], x[2], x[1]) for x in submesh.pos_verts]
    for i, p in enumerate(controlpoints):
        mesh.SetControlPointAt(p, i)
    for face in submesh.faces:
        mesh.BeginPolygon()
        mesh.AddPolygon(face[0]-1)
        mesh.AddPolygon(face[1]-1)
        mesh.AddPolygon(face[2]-1)
        mesh.EndPolygon()

    # node = fbx.FbxNode.Create(d2map.fbx_model.scene, name)
    # node.SetNodeAttribute(mesh)

    return mesh


def apply_diffuse(d2map, submesh, node):
    # print('applying diffuse', tex_name)
    lMaterialName = f'mat {submesh.material.name}'
    if lMaterialName in d2map.materials.keys():
        node.AddMaterial(d2map.materials[lMaterialName])
        return
    lMaterial = fbx.FbxSurfacePhong.Create(d2map.fbx_model.scene, lMaterialName)
    d2map.materials[lMaterialName] = lMaterial
    lMaterial.DiffuseFactor.Set(1)
    lMaterial.ShadingModel.Set('Phong')
    node.AddMaterial(lMaterial)


    gTexture = fbx.FbxFileTexture.Create(d2map.fbx_model.scene, f'Diffuse Texture {submesh.diffuse}')
    lTexPath = f'I:/maps/{d2map.pkg_name}_fbx/textures/{submesh.diffuse}.png'
    # print('tex path', f'C:/d2_maps/{folder_name}_fbx/textures/{tex_name}.png')
    gTexture.SetFileName(lTexPath)
    gTexture.SetTextureUse(fbx.FbxFileTexture.eStandard)
    gTexture.SetMappingType(fbx.FbxFileTexture.eUV)
    gTexture.SetMaterialUse(fbx.FbxFileTexture.eModelMaterial)
    gTexture.SetSwapUV(False)
    gTexture.SetTranslation(0.0, 0.0)
    gTexture.SetScale(1.0, 1.0)
    gTexture.SetRotation(0.0, 0.0)

    if lMaterial:
        lMaterial.Diffuse.ConnectSrcObject(gTexture)
    else:
        raise RuntimeError('Material broken somewhere')


def create_uv(mesh, name, submesh: met.Submesh, layer):
    uvDiffuseLayerElement = fbx.FbxLayerElementUV.Create(mesh, f'diffuseUV {name}')
    uvDiffuseLayerElement.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
    uvDiffuseLayerElement.SetReferenceMode(fbx.FbxLayerElement.eDirect)
    # mesh.InitTextureUV()
    for i, p in enumerate(submesh.uv_verts):
        uvDiffuseLayerElement.GetDirectArray().Add(fbx.FbxVector2(p[0], p[1]))
    layer.SetUVs(uvDiffuseLayerElement, fbx.FbxLayerElement.eTextureDiffuse)
    return layer


def write_fbx(d2map: Map):
    d2map.fbx_model.export(save_path=f'I:/maps/{d2map.pkg_name}_fbx/{d2map.name}_efficient.fbx', ascii_format=False)
    print(f'Wrote fbx of {d2map.name}')


def unpack_folder(pkg_name, unreal, shaders):
    entries_refid = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, RefID') if y == '0x13AD'}
    entries_refpkg = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, RefPKG') if y == '0x0004'}
    entries_size = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, FileSizeB')}
    file_names = sorted(entries_refid.keys(), key=lambda x: entries_size[x])
    for file_name in file_names:
        if file_name in entries_refpkg.keys():
            # a = [x.split('.')[0] for x in os.listdir('C:\d2_maps/orphaned_0932_fbx/')]
            # if file_name in [x.split('.')[0] for x in os.listdir(f'C:\d2_maps/{pkg_name}_fbx/')]:
            #     continue
            # if '07A0' not in file_name:
            #     continue
            print(f'Unpacking {file_name}')
            unpack_map(file_name, pkg_name, unreal, shaders)


if __name__ == '__main__':
    import os
    # WARNING THIS CURRENTLY DOES NOT OVERWRITE SHADER FILES THAT ARE ALREADY WRITTEN
    pkg_db.start_db_connection()
    all_file_info = {x[0]: dict(zip(['RefID', 'RefPKG', 'FileType'], x[1:])) for x in
                     pkg_db.get_entries_from_table('Everything', 'FileName, RefID, RefPKG, FileType')}
    unpack_folder('edz_02bc', unreal=False, shaders=False)

import struct
import pkg_db
from dataclasses import dataclass, fields
import numpy as np
import os.path
import fbx
import pyfbx as pfb
import gf
import image_decoder_new as imager

version = '2_9_2_1_all'


@dataclass
class Stride12Header:
    EntrySize: np.uint32 = np.uint32(0)
    StrideLength: np.uint32 = np.uint32(0)
    DeadBeef: np.uint32 = np.uint32(0)


@dataclass
class LODSubmeshEntry:
    Offset: np.uint32 = np.uint32(0)
    FacesLength: np.uint32 = np.uint32(0)
    SecondIndexRef: np.uint16 = np.uint16(0)
    EntryType: np.uint16 = np.uint16(0)


class File:
    def __init__(self, name=None, uid=None, pkg_name=None):
        self.name = name
        self.uid = uid
        self.pkg_name = pkg_name

    def get_file_from_uid(self):
        self.name = gf.get_file_from_hash(self.uid)
        return self.pkg_name

    def get_uid_from_file(self):
        self.uid = gf.get_hash_from_file(self.name)
        return self.pkg_name

    def get_pkg_name(self):
        self.pkg_name = gf.get_pkg_name(self.name)
        return self.pkg_name


class ModelFile(File):
    def __init__(self, uid):
        super(ModelFile, self).__init__(uid=uid)
        self.models = []
        self.material_files = []
        self.model_data_file = File(name=self.get_model_data_file())
        self.model_file_hex = ''
        self.model_data_hex = ''

    def get_model_data_file(self):
        self.get_file_from_uid()
        pkg_name = self.get_pkg_name()
        if not pkg_name:
            raise RuntimeError('Invalid model file given')
        self.model_file_hex = gf.get_hex_data(f'C:/d2_output/{pkg_name}/{self.name}.bin')
        model_data_hash = self.model_file_hex[16:24]
        return gf.get_file_from_hash(model_data_hash)


class Model:
    def __init__(self):
        self.submeshes = []
        self.pos_verts_file = None
        self.pos_verts = []
        self.uv_verts_file = None
        self.uv_verts = []
        self.faces_file = None
        self.faces = []


class Submesh:
    def __init__(self):
        self.pos_verts = []
        self.uv_verts = []
        self.faces = []
        self.material = None
        self.diffuse = None
        self.normal = None
        self.type = 0


class HeaderFile(File):
    def __init__(self, header=None):
        super().__init__()
        self.header = header

    def get_header(self):
        if self.header:
            print('Cannot get header as header already exists.')
            return
        else:
            if not self.name:
                self.name = gf.get_file_from_hash(self.uid)
            pkg_name = gf.get_pkg_name(self.name)
            header_hex = gf.get_hex_data(f'C:/d2_output/{pkg_name}/{self.name}.bin')
            return get_header(header_hex, Stride12Header())


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


def get_model(model_file_hash):
    print(f'Getting model {model_file_hash}.')
    pkg_db.start_db_connection()
    model_file = ModelFile(model_file_hash)
    model_file.get_model_data_file()
    get_model_data(model_file)
    get_submeshes(model_file)
    get_materials(model_file)
    max_vert_used = 0
    for i, model in enumerate(model_file.models):
        for j, submesh in enumerate(model.submeshes):
            if submesh.type == 769 or submesh.type == 770 or submesh.type == 778:
                submesh.faces, max_vert_used = adjust_faces_data(submesh.faces, max_vert_used)
                submesh.faces = shift_faces_down(submesh.faces)
                write_fbx(model_file, submesh, f'{model_file_hash}_0_{i}_{j}')



def get_model_data(model_file: ModelFile):
    get_model_files(model_file)
    for model in model_file.models:
        model.pos_verts = get_verts_data(model.pos_verts_file)
        model.uv_verts = get_verts_data(model.uv_verts_file)
        model.pos_verts = scale_and_repos_pos_verts(model.pos_verts, model_file)
        model.uv_verts = scale_and_repos_uv_verts(model.uv_verts, model_file)
        model.faces = get_faces_data(model.faces_file)
    print('')


def get_model_files(model_file: ModelFile):
    model_file.model_data_file.get_pkg_name()
    model_file.model_data_hex = gf.get_hex_data(f'C:/d2_output/{model_file.model_data_file.pkg_name}/{model_file.model_data_file.name}.bin')
    split_hex = model_file.model_data_hex.split('BD9F8080')[-1]
    model_count = int(gf.get_flipped_hex(split_hex[:4], 4), 16)
    relevant_hex = split_hex[32:]

    models = []
    for i in range(model_count):
        model = Model()
        faces_hash = gf.get_flipped_hex(relevant_hex[32*i:32*i+8], 8)
        pos_verts_file = gf.get_flipped_hex(relevant_hex[32*i+8:32*i+16], 8)
        uv_verts_file = gf.get_flipped_hex(relevant_hex[32*i+16:32*i+24], 8)
        for j, hsh in enumerate([faces_hash, pos_verts_file, uv_verts_file]):
            hf = HeaderFile()
            hf.uid = gf.get_flipped_hex(hsh, 8)
            hf.name = gf.get_file_from_hash(hf.uid)
            hf.pkg_name = gf.get_pkg_name(hf.name)
            if j == 0:
                model.faces_file = hf
            elif j == 1:
                hf.header = hf.get_header()
                model.pos_verts_file = hf
            elif j == 2:
                if not hf.pkg_name:
                    print('No UV file found')
                hf.header = hf.get_header()
                model.uv_verts_file = hf
        models.append(model)
    model_file.models = models


def get_submeshes(model_file: ModelFile):
    unk_entries_count = int(gf.get_flipped_hex(model_file.model_data_hex[80*2:80*2 + 8], 4), 16)
    unk_entries_offset = 96

    end_offset = unk_entries_offset + unk_entries_count * 8
    end_place = int(model_file.model_data_hex[end_offset*2:].find('BD9F8080')/2)
    submesh_entries_count = int(gf.get_flipped_hex(model_file.model_data_hex[(end_offset + end_place + 4)*2:(end_offset + end_place + 6)*2], 4), 16)
    submesh_entries_offset = end_offset + end_place + 20
    submesh_entries_length = submesh_entries_count * 12
    submesh_entries_hex = model_file.model_data_hex[submesh_entries_offset*2:submesh_entries_offset*2 + submesh_entries_length*2]
    submesh_entries = [submesh_entries_hex[i:i+24] for i in range(0, len(submesh_entries_hex), 24)]

    actual_submeshes = []
    for e in submesh_entries:
        entry = get_header(e, LODSubmeshEntry())
        if entry.EntryType == 769 or entry.EntryType == 770 or entry.EntryType == 778:
            actual_submeshes.append(entry)

    relevant_textures = get_materials(model_file)

    for i, e in enumerate(actual_submeshes):
        submesh = Submesh()
        model = model_file.models[e.SecondIndexRef]
        submesh.faces = model.faces[int(e.Offset/3):int((e.Offset + e.FacesLength)/3)]
        submesh.pos_verts = trim_verts_data(model.pos_verts, submesh.faces)
        submesh.uv_verts = trim_verts_data(model.uv_verts, submesh.faces)
        submesh.type = e.EntryType
        submesh.material = File(name=relevant_textures[i])
        model.submeshes.append(submesh)


def get_materials(model_file: ModelFile):
    texture_count = int(gf.get_flipped_hex(model_file.model_data_hex[80*2:84*2], 8), 16)
    texture_id_entries = [[int(gf.get_flipped_hex(model_file.model_data_hex[i:i+4], 4), 16), model_file.model_data_hex[i+4:i+8], model_file.model_data_hex[i+8:i+12]] for i in range(96*2, 96*2+texture_count*16, 16)]
    texture_entries = [model_file.model_file_hex[i:i+8] for i in range(176*2, 176*2+texture_count*8, 8)]
    relevant_textures = {}
    for i, entry in enumerate(texture_id_entries):
        # 0000 is LOD0, 0003 is prob LOD1, 000C LOD2
        if entry[1] == '0000':
            relevant_textures[entry[0]] = gf.get_file_from_hash(texture_entries[i])
    return relevant_textures


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


def adjust_faces_data(faces_data, max_vert_used):
    new_faces_data = []
    all_v = []
    for face in faces_data:
        for v in face:
            all_v.append(v)
    starting_face_number = min(all_v) -1
    all_v = []
    for face in faces_data:
        new_face = []
        for v in face:
            new_face.append(v - starting_face_number + max_vert_used)
            all_v.append(v - starting_face_number + max_vert_used)
        new_faces_data.append(new_face)
    return new_faces_data, max(all_v)


def scale_and_repos_pos_verts(verts_data, model_file):
    scale = struct.unpack('f', bytes.fromhex(model_file.model_file_hex[0x6C*2:0x6C*2 + 8]))[0]
    for i in range(len(verts_data)):
        for j in range(3):
            verts_data[i][j] *= scale

    position_shift = [struct.unpack('f', bytes.fromhex(model_file.model_file_hex[192 + 8 * i:192 + 8 * (i + 1)]))[0] for i in range(3)]
    for i in range(3):
        for j in range(len(verts_data)):
            verts_data[j][i] -= (scale - position_shift[i])
    return verts_data


def scale_and_repos_uv_verts(verts_data, model_file):
    scales = [struct.unpack('f', bytes.fromhex(model_file.model_file_hex[0x70*2+i*8:0x70*2+(i+1)*8]))[0] for i in range(2)]
    position_shifts = [struct.unpack('f', bytes.fromhex(model_file.model_file_hex[0x78*2+i*8:0x78*2+(i+1)*8]))[0] for i in range(2)]
    for i in range(len(verts_data)):
        verts_data[i][0] *= scales[0]
        verts_data[i][1] *= -scales[1]

    for j in range(len(verts_data)):
        verts_data[j][0] -= (scales[0] - position_shifts[0])
        verts_data[j][1] -= (scales[1] * position_shifts[0])/2

    return verts_data


def get_faces_data(faces_file):
    ref_file = f"{all_file_info[faces_file.name]['RefPKG'][2:]}-{all_file_info[faces_file.name]['RefID'][2:]}"
    ref_pkg_name = gf.get_pkg_name(ref_file)
    ref_file_type = all_file_info[ref_file]['FileType']
    faces = []
    if ref_file_type == "Faces Header":
        faces_hex = gf.get_hex_data(f'C:/d2_output/{ref_pkg_name}/{ref_file}.bin')
        int_faces_data = [int(gf.get_flipped_hex(faces_hex[i:i+4], 4), 16)+1 for i in range(0, len(faces_hex), 4)]
        for i in range(0, len(int_faces_data), 3):
            face = []
            for j in range(3):
                face.append(int_faces_data[i+j])
            faces.append(face)
        return faces
    else:
        print(f'Faces: Incorrect type of file {ref_file_type} for ref file {ref_file} verts file {faces_file}')
        return None


def get_float16(hex_data, j, is_uv=False):
    flt = get_signed_int(gf.get_flipped_hex(hex_data[j * 4:j * 4 + 4], 4), 16)
    # if j == 1 and is_uv:
    #     flt *= -1
    flt = 1 + flt / (2 ** 15 - 1)
    return flt


def get_signed_int(hexstr, bits):
    value = int(hexstr, 16)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value


def get_verts_data(verts_file):
    """
    Stride length 48 is a dynamic and physics-enabled object.
    """
    # TODO deal with this
    pkg_name = verts_file.pkg_name
    if not pkg_name:
        return None
    ref_file = f"{all_file_info[verts_file.name]['RefPKG'][2:]}-{all_file_info[verts_file.name]['RefID'][2:]}"
    ref_pkg_name = gf.get_pkg_name(ref_file)
    ref_file_type = all_file_info[ref_file]['FileType']
    if ref_file_type == "Stride Header":
        stride_header = verts_file.header

        stride_hex = gf.get_hex_data(f'C:/d2_output/{ref_pkg_name}/{ref_file}.bin')

        hex_data_split = [stride_hex[i:i + stride_header.StrideLength * 2] for i in
                          range(0, len(stride_hex), stride_header.StrideLength * 2)]
    else:
        print(f'Verts: Incorrect type of file {ref_file_type} for ref file {ref_file} verts file {verts_file}')
        return None

    if stride_header.StrideLength == 4:
        """
        UV info for dynamic, physics-based objects.
        """
        coords = get_coords_4(hex_data_split)
    elif stride_header.StrideLength == 8:
        """
        Coord info for static and dynamic, non-physics objects.
        ? info for dynamic, physics-based objects.
        """
        coords = get_coords_8(hex_data_split)
    elif stride_header.StrideLength == 12:
        """
        Coord info takes up same 8 stride, also 2 extra bits.
        """
        # TODO ADD PROPER SUPPORT
        coords = get_coords_8(hex_data_split)
    elif stride_header.StrideLength == 16:
        """
        """
        coords = get_coords_16(hex_data_split)
    elif stride_header.StrideLength == 20:
        """
        UV info for static and dynamic, non-physics objects.
        """
        coords = get_coords_20(hex_data_split)
    elif stride_header.StrideLength == 24:
        """
        UV info for dynamic, non-physics objects gear?
        """
        coords = get_coords_24(hex_data_split)
    elif stride_header.StrideLength == 28:
        """
        Coord info takes up same 8 stride, idk about other stuff
        """
        # TODO ADD PROPER SUPPORT
        coords = get_coords_8(hex_data_split)
    elif stride_header.StrideLength == 32:
        """
        Coord info takes up same 8 stride, idk about other stuff
        """
        # TODO ADD PROPER SUPPORT
        coords = get_coords_8(hex_data_split)
    elif stride_header.StrideLength == 48:
        """
        Coord info for dynamic, physics-based objects.
        """
        # print('Stride 48')
        coords = get_coords_48(hex_data_split)
    else:
        print(f'Need to add support for stride length {stride_header.StrideLength}, file is {verts_file.name} ref {ref_file}')
        quit()

    return coords


def get_coords_4(hex_data_split):
    coords = []
    for hex_data in hex_data_split:
        coord = []
        for j in range(2):
            flt = get_float16(hex_data, j, is_uv=True)
            coord.append(flt)
        coords.append(coord)
    return coords


def get_coords_8(hex_data_split):
    coords = []
    for hex_data in hex_data_split:
        coord = []
        for j in range(3):
            flt = get_float16(hex_data, j, is_uv=False)
            coord.append(flt)
        coords.append(coord)
    return coords


def get_coords_16(hex_data_split):
    coords = []
    for hex_data in hex_data_split:
        coord = []
        for j in range(2):
            flt = get_float16(hex_data, j, is_uv=False)
            coord.append(flt)
        coords.append(coord)
    return coords


def get_coords_20(hex_data_split):
    coords = []
    for hex_data in hex_data_split:
        coord = []
        for j in range(2):
            flt = get_float16(hex_data, j, is_uv=True)
            coord.append(flt)
        coords.append(coord)
    return coords


def get_coords_24(hex_data_split):
    coords = []
    for hex_data in hex_data_split:
        coord = []
        for j in range(2):
            flt = get_float16(hex_data, j, is_uv=True)
            coord.append(flt)
        coords.append(coord)
    return coords


def get_coords_48(hex_data_split):
    coords = []
    for hex_data in hex_data_split:
        coord = []
        for j in range(3):
            flt = struct.unpack('f', bytes.fromhex(hex_data[j * 8:j * 8 + 8]))[0]
            coord.append(flt)
        coords.append(coord)
    return coords


def trim_verts_data(verts_data, faces_data):
    all_v = []
    for face in faces_data:
        for v in face:
            all_v.append(v)
    return verts_data[min(all_v)-1:max(all_v)]


def write_fbx(model_file: ModelFile, submesh: Submesh, name):
    model = pfb.Model()
    model.create_node()
    get_submesh_textures(model_file, submesh)
    node, mesh = create_mesh(model, submesh.pos_verts, submesh.faces, name)
    if submesh.diffuse:
        layer = mesh.GetLayer(0)
        if not layer:
            mesh.CreateLayer()
        layer = mesh.GetLayer(0)

        node = apply_diffuse(model, submesh.diffuse, f'C:/d2_model_temp/texture_models/{model_file.uid}/textures/{submesh.diffuse}.png', node)

        layer = create_uv(mesh, submesh.diffuse, submesh.uv_verts, layer)
        node.SetShadingMode(fbx.FbxNode.eTextureShading)
    model.scene.GetRootNode().AddChild(node)

    gf.mkdir(f'C:/d2_model_temp/texture_models/{model_file.uid}/')
    model.export(save_path=f'C:/d2_model_temp/texture_models/{model_file.uid}/{name}.fbx', ascii_format=False)


def get_submesh_textures(model_file: ModelFile, submesh: Submesh, custom_dir=False):
    f_hex = gf.get_hex_data(f'C:/d2_output/{submesh.material.get_pkg_name()}/{submesh.material.name}.bin')
    offset = f_hex.find('11728080')
    count = int(gf.get_flipped_hex(f_hex[offset-16:offset-8], 8), 16)
    images = [gf.get_file_from_hash(f_hex[offset+16+8+8*(2*i):offset+16+8*(2*i)+16]) for i in range(count)]
    submesh.diffuse = images[0]
    for img in images:
        if custom_dir:
            gf.mkdir(f'{custom_dir}/')
            if not os.path.exists(f'{custom_dir}/{img}.png'):
                imager.get_image_from_file(f'C:/d2_output/{gf.get_pkg_name(img)}/{img}.bin', f'{custom_dir}/')
        else:
            gf.mkdir(f'C:/d2_model_temp/texture_models/{model_file.uid}/textures/')
            if not os.path.exists(f'C:/d2_model_temp/texture_models/{model_file.uid}/textures/{img}.png'):
                imager.get_image_from_file(f'C:/d2_output/{gf.get_pkg_name(img)}/{img}.bin', f'C:/d2_model_temp/texture_models/{model_file.uid}/textures/')



def create_mesh(fbx_map, pos_verts_data, faces_data, name):
    mesh = fbx.FbxMesh.Create(fbx_map.scene, name)
    controlpoints = [fbx.FbxVector4(x[0], x[1], x[2]) for x in pos_verts_data]
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

    return node, mesh


def apply_diffuse(fbx_map, tex_name, tex_path, node):
    # print('applying diffuse', tex_name)
    lMaterialName = f'mat {tex_name}'
    lMaterial = fbx.FbxSurfacePhong.Create(fbx_map.scene, lMaterialName)
    lMaterial.DiffuseFactor.Set(1)
    lMaterial.ShadingModel.Set('Phong')
    node.AddMaterial(lMaterial)


    gTexture = fbx.FbxFileTexture.Create(fbx_map.scene, f'Diffuse Texture {tex_name}')
    # lTexPath = f'C:/d2_maps/{folder_name}_fbx/textures/{tex_name}.png'
    lTexPath = tex_path
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
    return node


def create_uv(mesh, name, uv_verts_data, layer):
    uvDiffuseLayerElement = fbx.FbxLayerElementUV.Create(mesh, f'diffuseUV {name}')
    uvDiffuseLayerElement.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
    uvDiffuseLayerElement.SetReferenceMode(fbx.FbxLayerElement.eDirect)
    # mesh.InitTextureUV()
    for i, p in enumerate(uv_verts_data):
        uvDiffuseLayerElement.GetDirectArray().Add(fbx.FbxVector2(p[0], p[1]))
    layer.SetUVs(uvDiffuseLayerElement, fbx.FbxLayerElement.eTextureDiffuse)
    return layer


if __name__ == '__main__':
    pkg_db.start_db_connection()
    all_file_info = {x[0]: dict(zip(['RefID', 'RefPKG', 'FileType'], x[1:])) for x in
                     pkg_db.get_entries_from_table('Everything', 'FileName, RefID, RefPKG, FileType')}

    # 75465881
    # 74324081
    # 86BFFE80
    get_model('86BFFE80')
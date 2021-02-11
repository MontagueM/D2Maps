import struct
import pkg_db
from dataclasses import dataclass, fields
import numpy as np
import os.path
import fbx
import pyfbx as pfb
import gf
import image_decoder_new as imager  # Unreal only?
import image_extractor as imager  # Blender
import get_shader as shaders



@dataclass
class Stride12Header:
    EntrySize: np.uint32 = np.uint32(0)
    StrideLength: np.uint16 = np.uint16(0)
    VertsType: np.uint16 = np.uint16(0)
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
        self.fb = None

    def get_file_from_uid(self):
        self.name = gf.get_file_from_hash(self.uid)
        return self.name

    def get_uid_from_file(self):
        self.uid = gf.get_hash_from_file(self.name)
        return self.uid

    def get_pkg_name(self):
        self.pkg_name = gf.get_pkg_name(self.name)
        return self.pkg_name

    def get_fb(self):
        self.fb = open(f'I:/d2_output_3_1_0_0/{self.pkg_name}/{self.name}.bin', 'rb').read()


class ModelFile(File):
    def __init__(self, uid):
        super(ModelFile, self).__init__(uid=uid)
        self.models = []
        self.material_files = []
        self.materials = {}
        self.model_data_file = File(name=self.get_model_data_file())
        self.model_file_fb = b''
        self.model_data_fb = b''
        self.new_type = None

    def get_model_data_file(self):
        self.get_file_from_uid()
        pkg_name = self.get_pkg_name()
        if not pkg_name:
            return None
            # raise RuntimeError('Invalid model file given')
        self.model_file_fb = open(f'I:/d2_output_3_1_0_0/{pkg_name}/{self.name}.bin', 'rb').read()
        model_data_hash = self.model_file_fb[8:12]
        return gf.get_file_from_hash(model_data_hash.hex())


class Model:
    def __init__(self):
        self.submeshes = []
        self.pos_verts_file = None
        self.pos_verts = []
        self.extra_verts_file = None
        self.vert_colour_file = None
        self.uv_verts = []
        self.vertex_colour = []
        self.faces_file = None
        self.faces = []


class Submesh:
    def __init__(self):
        self.pos_verts = []
        self.adjusted_pos_verts = []
        self.uv_verts = []
        self.vertex_colour = []
        self.faces = []
        self.material = None
        self.textures = []
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
            header_fb = open(f'I:/d2_output_3_1_0_0/{pkg_name}/{self.name}.bin', 'rb').read()
            self.header = get_header(header_fb, Stride12Header())
            return self.header


def get_header(fb, header):
    # The header data is 0x16F bytes long, so we need to x2 as python reads each nibble not each byte

    for f in fields(header):
        if f.type == np.uint32:
            value = gf.get_uint32(fb, 0)
            setattr(header, f.name, value)
            fb = fb[4:]
        elif f.type == np.uint16:
            value = gf.get_uint16(fb, 0)
            setattr(header, f.name, value)
            fb = fb[2:]
    return header


def get_model(model_file_hash):
    print(f'Getting model {model_file_hash}.')
    # pkg_db.start_db_connection()
    model_file = ModelFile(model_file_hash)
    model_file.get_model_data_file()
    get_model_data(model_file, all_file_info)
    get_submeshes(model_file)
    # get_materials(model_file)
    max_vert_used = 0
    for i, model in enumerate(model_file.models):
        for j, submesh in enumerate(model.submeshes):
            if submesh.type == 769 or submesh.type == 770 or submesh.type == 778 or submesh.type == 'Decal':
                submesh.faces, max_vert_used = adjust_faces_data(submesh.faces, max_vert_used)
                submesh.faces = shift_faces_down(submesh.faces)
                export_fbx(model_file, submesh, f'{model_file_hash}_0_{i}_{j}')



def get_model_data(model_file: ModelFile, all_file_info):
    ret = get_model_files(model_file)
    if not ret:
        return
    for model in model_file.models:
        model.pos_verts = get_verts_data(model.pos_verts_file, all_file_info, is_uv=False)
        model.pos_verts = scale_and_repos_pos_verts(model.pos_verts, model_file)

        if model.extra_verts_file:
            coords = get_verts_data(model.extra_verts_file, all_file_info, is_uv=True)
            model.uv_verts = coords[0]
            model.uv_verts = scale_and_repos_uv_verts(model.uv_verts, model_file)

        if model.vert_colour_file:
            model.vertex_colour = get_vert_colour(model.vert_colour_file, all_file_info)
        model.faces = get_faces_data(model.faces_file, all_file_info)
    return True


def get_vert_colour(vc_file, all_file_info):
    vc_ref = f"{all_file_info[vc_file.name]['RefPKG'][2:]}-{all_file_info[vc_file.name]['RefID'][2:]}"
    fb = open(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(vc_ref)}/{vc_ref}.bin', 'rb').read()

    vert_colours = []
    for i in range(0, len(fb), 4):
        vc = [x/255 for x in fb[i:i+4]]
        vert_colours.append(vc)
        # if vcs[-1] != 1.0:  # Dyn only??
        #     vcs = [0, 0, 0, 0]
    if not any([x != [0, 0, 0, 0] for x in vert_colours]):
        vert_colours = []
    return vert_colours

def get_model_files(model_file: ModelFile):
    # Also due to new type
    if not model_file.model_data_file.name or model_file.model_data_file.name == '0400-0008':
        print('NEW TYPEa')
        return
    model_file.model_data_file.get_pkg_name()
    model_file.model_data_fb = open(f'I:/d2_output_3_1_0_0/{model_file.model_data_file.pkg_name}/{model_file.model_data_file.name}.bin', 'rb').read()
    split_fb = model_file.model_data_fb.split(b'\xB8\x9F\x80\x80')[-1]
    model_count = gf.get_uint16(split_fb, 0)
    relevant_fb = split_fb[16:]

    models = []
    for i in range(model_count):
        model = Model()
        new_type = False
        # This is due to a new type of model/dynamics being added, ignoring for now
        # if faces_hash == '00000000' or pos_verts_file == '00000000':
        #     print('NEW TYPEb')
        #     return
        # Some normal statics also include this table???
        if b'\x2F\x6D\x80\x80' in model_file.model_file_fb and b'\x14\x00\x80\x80' not in model_file.model_file_fb:
            model_file.new_type = True
            # Pretty sure this table only holds a single model file, so need to break after this.
            # This table is more complicated (look at onenote) but this is all I think I need for it to work
            print('New static type')
            offset = model_file.model_file_fb.find(b'\x2F\x6D\x80\x80')+8
            if offset == -1:
                raise Exception('BROKEN NEW TYPE')
            count = gf.get_uint32(model_file.model_file_fb, 4)  # I need count here as I think some files can have four
            if count != 3:
                print(f'Count != 3 (== {count})')
            faces_hash = model_file.model_file_fb[offset+8:offset+12]
            pos_verts_file = model_file.model_file_fb[offset+12:offset+16]
            uv_verts_file = model_file.model_file_fb[offset+16:offset+20]
            vert_colour_file = None
            if pos_verts_file == '' or faces_hash == '':
                return

        else:
            # Normal static type
            faces_hash = relevant_fb[16 * i:16 * i + 4]
            pos_verts_file = relevant_fb[16 * i + 4:16 * i + 8]
            uv_verts_file = relevant_fb[16 * i + 8:16 * i + 12]
            vert_colour_file = relevant_fb[16 * i + 12:16 * i + 16]
            if pos_verts_file == '' or faces_hash == '':
                return
        for j, hsh in enumerate([faces_hash, pos_verts_file, uv_verts_file, vert_colour_file]):
            if not hsh:
                continue
            hf = HeaderFile()
            hf.uid = hsh.hex()
            hf.name = hf.get_file_from_uid()
            hf.pkg_name = hf.get_pkg_name()
            if j == 0:
                model.faces_file = hf
            elif j == 1:
                hf.header = hf.get_header()
                model.pos_verts_file = hf
            elif j == 2:
                if not hf.pkg_name:
                    print('No extra verts file found')
                else:
                    hf.header = hf.get_header()
                    model.extra_verts_file = hf
            elif j == 3:
                # if not hf.pkg_name:
                #     print('No vert colour file found')
                # else:
                if vert_colour_file and vert_colour_file != b'\xFF\xFF\xFF\xFF':
                    hf.header = hf.get_header()
                    model.vert_colour_file = hf
        models.append(model)

    model_file.models = models
    return True


def get_submeshes(model_file: ModelFile):
    unk_entries_count = gf.get_uint32(model_file.model_file_fb, 128)
    unk_entries_offset = 144

    end_offset = unk_entries_offset + unk_entries_count * 6
    end_place = model_file.model_data_fb[end_offset:].find(b'\xB8\x9F\x80\x80')
    submesh_entries_count = gf.get_uint16(model_file.model_data_fb, end_offset + end_place + 4)
    submesh_entries_offset = end_offset + end_place + 20
    submesh_entries_length = submesh_entries_count * 12
    submesh_entries_fb = model_file.model_data_fb[submesh_entries_offset:submesh_entries_offset + submesh_entries_length]
    submesh_entries = [submesh_entries_fb[i:i+12] for i in range(0, len(submesh_entries_fb), 12)]

    actual_submeshes = []
    for e in submesh_entries:
        entry = get_header(e, LODSubmeshEntry())
        actual_submeshes.append(entry)

    # TODO fix this double call
    relevant_textures = get_materials(model_file)

    for i, e in enumerate(actual_submeshes):
        submesh = Submesh()
        model = model_file.models[e.SecondIndexRef]
        submesh.faces = model.faces[int(e.Offset/3):int((e.Offset + e.FacesLength)/3)]
        submesh.pos_verts = trim_verts_data(model.pos_verts, submesh.faces)
        submesh.uv_verts = trim_verts_data(model.uv_verts, submesh.faces)
        submesh.vertex_colour = trim_verts_data(model.vertex_colour, submesh.faces)
        submesh.type = e.EntryType
        if i in relevant_textures.keys():
            submesh.material = File(name=relevant_textures[i])
            model.submeshes.append(submesh)

    # Decals
    submesh = Submesh()
    model = model_file.models[0]
    offset = model_file.model_file_fb.find(b'\x2F\x6D\x80\x80')+8
    if offset == 7:
        return
    faces_offset = gf.get_uint32(model_file.model_file_fb, offset+24)
    faces_length = gf.get_uint32(model_file.model_file_fb, offset+28)
    submesh.faces = model.faces[int(faces_offset / 3):int((faces_offset + faces_length) / 3)]
    submesh.pos_verts = trim_verts_data(model.pos_verts, submesh.faces)
    submesh.uv_verts = trim_verts_data(model.uv_verts, submesh.faces)
    submesh.vertex_colour = trim_verts_data(model.vertex_colour, submesh.faces)
    submesh.material = File(name=gf.get_file_from_hash(model_file.model_file_fb[offset+0x20:offset+0x24].hex()))
    submesh.type = 'Decal'
    model.submeshes.append(submesh)


def get_materials(model_file: ModelFile):
    texture_count = gf.get_uint32(model_file.model_data_fb, 128)
    texture_id_entries = [[gf.get_uint16(model_file.model_data_fb, i), model_file.model_data_fb[i+2:i+3], model_file.model_data_fb[i+4:i+6]] for i in range(144, 144+texture_count*6, 6)]
    texture_entries = [model_file.model_file_fb[i:i+4] for i in range(144, 144+texture_count*4, 4)]
    relevant_textures = {}
    for i, entry in enumerate(texture_id_entries):
        # 0000 is LOD0, 0003 is prob LOD1, 000C LOD2
        if entry[1] == b'\x00':
            relevant_textures[entry[0]] = gf.get_file_from_hash(texture_entries[i].hex())
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
    scale = struct.unpack('f', model_file.model_data_fb[92:92 + 4])[0]
    for i in range(len(verts_data)):
        for j in range(3):
            verts_data[i][j] *= scale

    position_shift = [struct.unpack('f', model_file.model_data_fb[80 + 4 * i:80 + 4 * (i + 1)])[0] for i in range(3)]
    for i in range(3):
        for j in range(len(verts_data)):
            verts_data[j][i] -= (scale - position_shift[i])
    return verts_data


def scale_and_repos_uv_verts(verts_data, model_file):
    # return verts_data
    scales = [struct.unpack('f', model_file.model_data_fb[0x60+i*4:0x60+(i+1)*4])[0] for i in range(2)]
    position_shifts = [struct.unpack('f', model_file.model_data_fb[0x68+i*4:0x68+(i+1)*4])[0] for i in range(2)]
    for i in range(len(verts_data)):
        verts_data[i][0] *= scales[0]
        verts_data[i][1] *= -scales[1]

    for j in range(len(verts_data)):
        verts_data[j][0] -= (scales[0] - position_shifts[0])
        verts_data[j][1] += (scales[1] - position_shifts[1] + 1)

    # flip uv tests

    # for j in range(len(verts_data)):
    #     verts_data[j] = verts_data[j][::-1]

    return verts_data


def get_faces_data(faces_file, all_file_info):
    faces_file.get_file_from_uid()
    ref_file = f"{all_file_info[faces_file.name]['RefPKG'][2:]}-{all_file_info[faces_file.name]['RefID'][2:]}"
    ref_pkg_name = gf.get_pkg_name(ref_file)
    ref_file_type = all_file_info[ref_file]['FileType']
    faces = []
    if ref_file_type == "Faces Data":
        faces_fb = open(f'I:/d2_output_3_1_0_0/{ref_pkg_name}/{ref_file}.bin', 'rb').read()
        int_faces_data = [gf.get_uint16(faces_fb, i)+1 for i in range(0, len(faces_fb), 2)]
        for i in range(0, len(int_faces_data), 3):
            face = []
            for j in range(3):
                face.append(int_faces_data[i+j])
            faces.append(face)
        return faces
    else:
        print(f'Faces: Incorrect type of file {ref_file_type} for ref file {ref_file} verts file {faces_file}')
        return None


def get_float16(fb, j):
    flt = int.from_bytes(fb[j * 2:j * 2 + 2], 'little', signed=True)
    flt = 1 + flt / (2 ** 15 - 1)
    return flt


def get_verts_data(verts_file, all_file_info, is_uv):
    """
    Stride length 48 is a dynamic and physics-enabled object.
    """
    # TODO deal with this
    pkg_name = verts_file.get_pkg_name()
    if not pkg_name:
        raise Exception('No pkg name')
    ref_file = f"{all_file_info[verts_file.name]['RefPKG'][2:]}-{all_file_info[verts_file.name]['RefID'][2:]}"
    ref_pkg_name = gf.get_pkg_name(ref_file)
    ref_file_type = all_file_info[ref_file]['FileType']
    if ref_file_type == "Vertex Data":
        stride_header = verts_file.header

        stride_fb = open(f'I:/d2_output_3_1_0_0/{ref_pkg_name}/{ref_file}.bin', 'rb').read()

        fb_data_split = [stride_fb[i:i + stride_header.StrideLength] for i in
                          range(0, len(stride_fb), stride_header.StrideLength)]
    else:
        print(f'Verts: Incorrect type of file {ref_file_type} for ref file {ref_file} verts file {verts_file}')
        return None

    # print('stridelength', stride_header.StrideLength)
    if stride_header.StrideLength == 4:
        """
        UV info for dynamic, physics-based objects.
        """
        coords = get_coords_4(fb_data_split)
    elif stride_header.StrideLength == 8:
        """
        Coord info for static and dynamic, non-physics objects.
        ? info for dynamic, physics-based objects.
        Can also be uv for some reason
        """
        coords = get_coords_8(fb_data_split, is_uv)
    elif stride_header.StrideLength == 12:
        """
        Coord info takes up same 8 stride, also 2 extra bits.
        Also UV sometimes? 2 extra bits will be for UV
        """
        # TODO ADD PROPER SUPPORT
        coords = get_coords_12(fb_data_split, is_uv)
    elif stride_header.StrideLength == 16:
        """
        UV
        """
        coords = get_coords_16(fb_data_split, is_uv)
    elif stride_header.StrideLength == 20:
        """
        UV info for static and dynamic, non-physics objects.
        """
        coords = get_coords_20(fb_data_split)
    elif stride_header.StrideLength == 24:
        """
        UV info for dynamic, non-physics objects gear?
        """
        coords = get_coords_24(fb_data_split)
    elif stride_header.StrideLength == 28:
        """
        Coord info takes up same 8 stride, idk about other stuff
        """
        # TODO ADD PROPER SUPPORT
        coords = get_coords_8(fb_data_split)
    elif stride_header.StrideLength == 32:
        """
        Coord info takes up same 8 stride, idk about other stuff
        """
        # TODO ADD PROPER SUPPORT
        coords = get_coords_8(fb_data_split)
    elif stride_header.StrideLength == 48:
        """
        Coord info for dynamic, physics-based objects.
        """
        # print('Stride 48')
        coords = get_coords_48(fb_data_split)
    else:
        print(f'Need to add support for stride length {stride_header.StrideLength}, file is {verts_file.name} ref {ref_file}')
        quit()

    return coords


def get_coords_4(fb_data_split):
    coords = []
    for fb_data in fb_data_split:
        coord = []
        for j in range(2):
            flt = get_float16(fb_data, j)
            coord.append(flt)
        coords.append(coord)
    return [coords, []]


def get_coords_8(fb_data_split, is_uv=False):
    if is_uv:
        uvs = []
        for fb_data in fb_data_split:
            uv = []
            for j in range(4):
                flt = get_float16(fb_data, j)
                if j % 2 == 0:
                    uv.append(flt)
            uvs.append(uv)
        return [uvs, []]
    else:
        coords = []
        for fb_data in fb_data_split:
            coord = []
            for j in range(3):
                flt = get_float16(fb_data, j)
                coord.append(flt)
            coords.append(coord)
        return coords


def get_coords_12(fb_data_split, is_uv):
    uvs = []
    coords = []
    for fb_data in fb_data_split:
        if is_uv:
            uv = []
            for j in range(0, 2):
                flt = get_float16(fb_data, j)
                uv.append(flt)
            uvs.append(uv)
        else:
            coord = []
            for j in range(3):
                flt = get_float16(fb_data, j)
                coord.append(flt)
            coords.append(coord)
    if is_uv:
        return [uvs, []]
    else:
        return coords


def get_coords_16(fb_data_split, is_uv):
    uvs = []
    coords = []
    for fb_data in fb_data_split:
        if is_uv:
            uv = []
            for j in range(2):
                flt = get_float16(fb_data, j)
                uv.append(flt)
            uvs.append(uv)
        else:
            coord = []
            for j in range(3):
                flt = get_float16(fb_data, j)
                coord.append(flt)
            coords.append(coord)
    if is_uv:
        return [uvs, []]
    else:
        return coords


def get_coords_20(fb_data_split):
    uvs = []
    normals = []
    for fb_data in fb_data_split:
        uv = []
        norm = []
        for j in range(2):
            flt = get_float16(fb_data, j)
            uv.append(flt)
        uvs.append(uv)
        for j in range(2, 5):
            flt = get_float16(fb_data, j)-1
            norm.append(flt)
        normals.append(norm)
    return [uvs, normals]


def get_coords_24(fb_data_split):
    coords = []
    for fb_data in fb_data_split:
        coord = []
        for j in range(2):
            flt = get_float16(fb_data, j)
            coord.append(flt)
        coords.append(coord)
    return [coords, []]


def get_coords_48(fb_data_split):
    coords = []
    for fb_data in fb_data_split:
        coord = []
        for j in range(3):
            flt = struct.unpack('f', bytes.fromhex(fb_data[j * 4:j * 4 + 4]))[0]
            coord.append(flt)
        coords.append(coord)
    return coords


def trim_verts_data(verts_data, faces_data):
    all_v = []
    for face in faces_data:
        for v in face:
            all_v.append(v)
    return verts_data[min(all_v)-1:max(all_v)]


def export_fbx(model_file: ModelFile, submesh: Submesh, name):
    gf.mkdir(f'I:/static_models/{model_file.uid}/')

    model = pfb.Model()
    node, mesh = create_mesh(model, submesh.pos_verts, submesh.faces, name)
    # Disabled materials for now
    if not mesh.GetLayer(0):
        mesh.CreateLayer()
    layer = mesh.GetLayer(0)

    if submesh.material:
        get_submesh_textures(model_file, submesh, hash64_table, all_file_info)
        # shaders.get_shader(model_file, submesh, all_file_info, name)
        # apply_shader(model, model_file, submesh, node)
        print(f'submesh {name} has mat file {submesh.material.name} with textures {submesh.textures}')
        if submesh.diffuse:
            apply_diffuse(model, submesh.diffuse, f'I:/static_models/{model_file.uid}/textures/{submesh.diffuse}.png', node)
            # set_normals(mesh, submesh.diffuse, submesh.norm_verts, layer)
            node.SetShadingMode(fbx.FbxNode.eTextureShading)

    if submesh.uv_verts:
        create_uv(mesh, submesh.diffuse, submesh.uv_verts, layer)
    if submesh.vertex_colour:
        add_vert_colours(mesh, model_file, submesh, layer)

    model.scene.GetRootNode().AddChild(node)

    # Disabled shaders for now
    # if True:
    # get_shader_info(model_file)

    model.export(save_path=f'I:/static_models/{model_file.uid}/{name}.fbx', ascii_format=False)
    print('Exported')


def add_vert_colours(mesh, name, submesh: Submesh, layer):
    vertColourElement = fbx.FbxLayerElementVertexColor.Create(mesh, f'colour')
    vertColourElement.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
    vertColourElement.SetReferenceMode(fbx.FbxLayerElement.eDirect)
    # mesh.InitTextureUV()
    for i, p in enumerate(submesh.vertex_colour):
        # vertColourElement.GetDirectArray().Add(fbx.FbxColor(p[0], p[1], p[2], 1))
        vertColourElement.GetDirectArray().Add(fbx.FbxColor(p[0], p[1], p[2], p[3]))

    layer.SetVertexColors(vertColourElement)


def get_shader_info(model_file):
    for material in model_file.material_files:
        if material.name == '0222-0CAF':
            print('')
        cbuffer_offsets, texture_offset = get_mat_tables(material)
        if not cbuffer_offsets:
            return
        textures = get_material_textures(material, texture_offset, hash64_table, all_file_info, custom_dir=f'I:/static_models/{model_file.uid}/textures/')
        get_shader_file(material, textures, cbuffer_offsets, all_file_info, custom_dir=f'I:/static_models/{model_file.uid}/shaders/')


def apply_shader(model, model_file, submesh: Submesh, node):
    lMaterialName = f'{submesh.material.name}'
    if lMaterialName in model_file.materials.keys():
        node.AddMaterial(model_file.materials[lMaterialName])
        return
    lMaterial = fbx.FbxSurfacePhong.Create(model.scene, lMaterialName)
    model_file.materials[lMaterialName] = lMaterial
    model_file.material_files.append(File(name=lMaterialName))
    # lMaterial.DiffuseFactor.Set(1)
    lMaterial.ShadingModel.Set('Phong')
    node.AddMaterial(lMaterial)


def get_submesh_textures(model_file: ModelFile, submesh: Submesh, hash64_table, all_file_info, custom_dir=False):
    if submesh.material.uid == 'FFFFFFFF' or submesh.material.name == 'FBFF-1FFF':
        return
    submesh.material.get_pkg_name()
    submesh.material.get_fb()
    offset = submesh.material.fb.find(b'\xCF\x6D\x80\x80')
    count = gf.get_uint32(submesh.material.fb, offset-8)
    # Arbritrary
    if count < 0 or count > 100:
        return
    # image_indices = [gf.get_file_from_hash(submesh.material.fhex[offset+16+8*(2*i):offset+16+8*(2*i)+8]) for i in range(count)]
    if b'\xFF\xFF\xFF\xFF' in submesh.material.fb[offset + 0x10:offset + 0x10 + 24 * count]:  # Uses new texture system
        images = [x for x in [gf.get_file_from_hash(
            hash64_table[submesh.material.fb[offset + 8 + 0x10 + 24 * i:offset + 8 + 0x10 + 24 * i + 8].hex().upper()])
                                        for i in range(count)] if x != 'FBFF-1FFF']
    else:
        images = [x for x in [
            gf.get_file_from_hash(submesh.material.fb[offset + 0x10 + 24 * i:offset + 0x10 + 24 * i + 4].hex().upper())
            for i in range(count)] if x != 'FBFF-1FFF']
    # _, images = zip(*sorted(zip(image_indices, images)))
    if len(images) == 0:
        return
    submesh.diffuse = images[0]
    submesh.textures = images
    for img in images:
        if custom_dir:
            gf.mkdir(f'{custom_dir}/')
            if not os.path.exists(f'{custom_dir}/{img}.png'):
                if img == 'FBFF-1FFF':
                    continue
                imager.get_image_from_file(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(img)}/{img}.bin', all_file_info, f'{custom_dir}/')
        else:
            gf.mkdir(f'I:/static_models/{model_file.uid}/textures/')
            if not os.path.exists(f'I:/static_models/{model_file.uid}/textures/{img}.png'):
                imager.get_image_from_file(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(img)}/{img}.bin', all_file_info, f'I:/static_models/{model_file.uid}/textures/')


def get_mat_tables(material):
    if material.uid == 'FFFFFFFF' or material.name == 'FBFF-1FFF':
        return None, None
    material.get_pkg_name()
    material.get_fb()
    cbuffers = []
    textures = -1

    texture_offset = 0x2A8
    table_offset = texture_offset + gf.get_uint32(material.fb, texture_offset)
    # table_count = int(gf.get_flipped_hex(material.fhex[table_offset:table_offset+8], 8), 16)
    table_type = material.fb[table_offset + 8:table_offset + 12]
    if table_type == b'\xCF\x6D\x80\x80':
        textures = table_offset

    start_offset = 0x2C0
    for i in range(6):
        current_offset = start_offset + 16*i
        table_offset = current_offset + gf.get_uint32(material.fb, current_offset)
        # table_count = int(gf.get_flipped_hex(material.fhex[table_offset:table_offset+8], 8), 16)
        table_type = material.fb[table_offset+8:table_offset+12]
        if table_type == b'\x90\x00\x80\x80':
            cbuffers.append(table_offset)

    # if textures == -1:
    #     raise Exception('Texture offset incorrect')

    return cbuffers, textures


def get_material_textures(material, texture_offset, hash64_table, all_file_info, custom_dir):
    if material.uid == 'FFFFFFFF' or material.name == 'FBFF-1FFF':
        return [], []
    material.get_pkg_name()
    material.get_fb()
    texture_offset += 8
    if texture_offset == 15:
        return [], []
    count = gf.get_uint32(material.fb, texture_offset-8)
    # Arbritrary
    if count < 0 or count > 100:
        return [], []
    image_indices = [material.fb[texture_offset+8+24*i] for i in range(count)]
    images = [gf.get_file_from_hash(hash64_table[material.fb[texture_offset+8+0x10+24*i:texture_offset+8+0x10+24*i+8].hex().upper()]) for i in range(count)]
    if len(images) == 0:
        return [], []
    for img in images:
        if custom_dir:
            gf.mkdir(f'{custom_dir}/')
            if not os.path.exists(f'{custom_dir}/{img}.png'):
                if img == 'FBFF-1FFF':
                    continue
                imager.get_image_from_file(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(img)}/{img}.bin', all_file_info, f'{custom_dir}/')
        # else:
        #     gf.mkdir(f'C:/d2_model_temp/texture_models/{model_file.uid}/textures/')
        #     if not os.path.exists(f'C:/d2_model_temp/texture_models/{model_file.uid}/textures/{img}.png'):
        #         imager.get_image_from_file(f'I:/d2_output_3_0_0_2/{gf.get_pkg_name(img)}/{img}.bin', f'C:/d2_model_temp/texture_models/{model_file.uid}/textures/')
    return images, image_indices


def get_shader_file(material, textures, indices, cbuffer_offsets, all_file_info, custom_dir):
    shaders.get_shader_from_mat(material, textures, indices, cbuffer_offsets, all_file_info, custom_dir)


def create_mesh(fbx_map, pos_verts_data, faces_data, name):
    mesh = fbx.FbxMesh.Create(fbx_map.scene, name)
    controlpoints = [fbx.FbxVector4(-x[0]*100, x[2]*100, x[1]*100) for x in pos_verts_data]
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
    gTexture.SetRelativeFileName(lTexPath)
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


def set_normals(mesh, name, normal_verts_data, layer):
    normalLayerElement = fbx.FbxLayerElementNormal.Create(mesh, f'normals {name}')
    normalLayerElement.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
    normalLayerElement.SetReferenceMode(fbx.FbxLayerElement.eDirect)
    for i, p in enumerate(normal_verts_data):
        normalLayerElement.GetDirectArray().Add(fbx.FbxVector4(p[0], p[1], p[2]))
    layer.SetNormals(normalLayerElement)


def create_uv(mesh, name, uv_verts_data, layer):
    uvDiffuseLayerElement = fbx.FbxLayerElementUV.Create(mesh, f'diffuseUV {name}')
    uvDiffuseLayerElement.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
    uvDiffuseLayerElement.SetReferenceMode(fbx.FbxLayerElement.eDirect)
    for i, p in enumerate(uv_verts_data):
        uvDiffuseLayerElement.GetDirectArray().Add(fbx.FbxVector2(p[0], p[1]))
    layer.SetUVs(uvDiffuseLayerElement, fbx.FbxLayerElement.eTextureDiffuse)


if __name__ == '__main__':
    version = '3_1_0_0'

    pkg_db.start_db_connection(f'I:/d2_pkg_db/{version}.db')
    all_file_info = {x[0]: dict(zip(['RefID', 'RefPKG', 'FileType'], x[1:])) for x in
                     pkg_db.get_entries_from_table('Everything', 'FileName, RefID, RefPKG, FileType')}

    pkg_db.start_db_connection(f'I:/d2_pkg_db/hash64/{version}.db')
    hash64_table = {x: y for x, y in pkg_db.get_entries_from_table('Everything', 'Hash64, Reference')}
    hash64_table['0000000000000000'] = 'FFFFFFFF'

    get_model('0851DC80')

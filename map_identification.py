import pkg_db
import gf
import struct


def from_location_bubble(bubble_name):
    if bubble_name not in loc_table:
        raise KeyError('Bubble is not in the location table')
    print(f'Map files related to {bubble_name}:')
    bubble_file = gf.get_file_from_hash(loc_table[bubble_name])
    get_bubble(bubble_file)


def from_text_string():
    pass


def get_bubble(bubble_file):
    fb = open(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(bubble_file)}/{bubble_file}.bin', 'rb').read()
    lower_file = gf.get_file_from_hash(fb[0x8:0xC].hex())
    fb = open(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(lower_file)}/{lower_file}.bin', 'rb').read()
    count = gf.get_uint32(fb, 0x8)
    # print(count)
    all_top_maps = [gf.get_file_from_hash(hash64_table[fb[i+0x8:i+0x10].hex().upper()]) for i in range(0x30, 0x30+count*0x10, 0x10) if b'\xFF\xFF\xFF\xFF' in fb[i:i+0x10]]
    for x in all_top_maps:
        fb = open(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(x)}/{x}.bin', 'rb').read()
        f = gf.get_file_from_hash(fb[0x54:0x58].hex())
        fb = open(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(f)}/{f}.bin', 'rb').read()
        if len(fb) < 0xD8:
            continue
        if gf.get_file_from_hash(fb[0xD8:0xDC].hex().upper()) not in all_file_info.keys():
            continue
        f = gf.get_file_from_hash(fb[0xD8:0xDC].hex())
        fb = open(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(f)}/{f}.bin', 'rb').read()
        map_file = gf.get_file_from_hash(fb[0x8:0xC].hex())
        fb = open(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(map_file)}/{map_file}.bin', 'rb').read()
        flt = [struct.unpack('f', fb[0x80+4*i:0x80+4*(i+1)])[0] for i in range(3)]

        print(x, map_file, len(fb), f'xyz {flt}')


def from_class_1E898080(class_array):
    for f in class_array:
        if '-' not in f:
            f = gf.get_file_from_hash(f)
        fb = open(f'I:/d2_output_3_1_0_0/{gf.get_pkg_name(f)}/{f}.bin', 'rb').read()
        print(f'Getting class for region {strinfo[fb[0x18:0x1C].hex().upper()]}')


if __name__ == '__main__':
    version = '3_1_0_0'
    pkg_db.start_db_connection(version=f'C:/Users\monta\OneDrive\Destiny 2 Datamining\TextExtractor\db/{version}.db')
    strinfo = {x: y for x, y in pkg_db.get_entries_from_table('Everything', 'Hash, String')}

    pkg_db.start_db_connection(f'I:/d2_pkg_db/hash64/{version}.db')
    hash64_table = {x: y for x, y in pkg_db.get_entries_from_table('Everything', 'Hash64, Reference')}

    pkg_db.start_db_connection(f'I:/d2_pkg_db/locations/{version}.db')
    loc_table = {x: y for x, y in pkg_db.get_entries_from_table('Everything', 'String, Hash')}


    pkg_db.start_db_connection(f'I:/d2_pkg_db/{version}.db')
    all_file_info = {x[0]: dict(zip(['Reference', 'FileType'], x[1:])) for x in
                     pkg_db.get_entries_from_table('Everything', 'FileName, Reference, FileType')}

    from_location_bubble('bubble|tangled_shore|plains')
    # get_bubble('0215-0B5B')
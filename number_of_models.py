import pkg_db
import gf

def number_of_models(pkg_name):
    pkg_db.start_db_connection()
    entries_refid = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, RefID') if y == '0x166D'}
    entries_refpkg = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, RefPKG') if y == '0x0004'}
    entries_size = {x: y for x, y in pkg_db.get_entries_from_table(pkg_name, 'FileName, FileSizeB')}
    file_names = sorted(entries_refid.keys(), key=lambda x: entries_size[x])
    for file_name in file_names:
        if file_name in entries_refpkg.keys():
            f = gf.File(name=file_name)
            f.get_hex_data()
            copycount = int(gf.get_flipped_hex(f.fhex[0x38*2:0x38*2+8], 8), 16)
            models = int(gf.get_flipped_hex(f.fhex[0x58*2:0x58*2+8], 8), 16)
            print(f'{file_name} | cc {copycount} | models {models}')


if __name__ == '__main__':
    number_of_models('city_tower_d2_0369')

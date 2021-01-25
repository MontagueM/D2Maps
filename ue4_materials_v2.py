import os
import unreal
from shutil import copyfile

"""
- Identify all materials in the material directory
- Ensure they have shader counterparts for _o0 1 2
- copy the EStackTemplate material and rename overwrite the material so we don't have to bother with mesh reassignment
- modify the new material to add in the textures and the modified custom exprs
"""


def get_valid_materials_textures():
    mats_texs = {}
    for asset in unreal.EditorAssetLibrary.list_assets(game_path + material_path, recursive=False):
        asset_data = unreal.EditorAssetLibrary.find_asset_data(asset)
        if asset_data.asset_class != "Material":
            continue

        matname = unreal.Paths.get_base_filename(asset)

        # if matname != 'tests':
        #     continue

        usf = asset.split('.')[-1].split('_')[0] + '_o0.usf'
        if 'FBFF' in usf:
            continue
        try:
            with open(top_path + shader_path + '/shaders/' + usf) as f:
                f = f.readlines()
                textures = f[1].split("', '")
                textures[0] = textures[0][4:]
                textures[-1] = textures[-1][:-3]
        except FileNotFoundError:
            print('Material has no textures')
            continue
        matpath = matname
        mats_texs[matpath] = textures
    return mats_texs


def copy_retarget_materials(mats_texs):
    for material in mats_texs.keys():
        print(top_path + material_path + material + '.uasset')
        unreal.EditorAssetLibrary.duplicate_asset(estack_template, game_path + material_path + material + '_estack')
        original_asset = unreal.EditorAssetLibrary.load_asset(game_path + material_path + material)
        replacement_asset = unreal.EditorAssetLibrary.load_asset(game_path + material_path + material + '_estack')
        unreal.EditorAssetLibrary.consolidate_assets(replacement_asset, [original_asset])
        # replacements.append(replacement_asset)


def modify_new_materials(mats_texs):
    for material, texs in mats_texs.items():
        mat = unreal.load_asset(game_path + material_path + material + '_estack')

        texsamples = add_tex_samples(mat, texs)

        it = unreal.ObjectIterator()
        for x in it:
            if x.get_outer() == mat:
                if isinstance(x, unreal.MaterialExpressionCustom):
                    print('Got custom expr')

                    # Modifying the custom expression node
                    desc = x.get_editor_property('desc')
                    if 'RT' in desc:
                        code = '#include "' + top_path + shader_path + '/shaders/' + material.split('_estack')[0] + '_o' + desc[2] + '.usf"\nreturn 0;'
                        inputs = []
                        for i in range(len(mats_texs[material])):
                            ci = unreal.CustomInput()
                            ci.set_editor_property('input_name', 't' + str(i))
                            inputs.append(ci)
                        ci = unreal.CustomInput()
                        ci.set_editor_property('input_name', 'tx')
                        inputs.append(ci)
                        x.set_editor_property('code', code)
                        x.set_editor_property('inputs', inputs)
                        x.set_editor_property('output_type', unreal.CustomMaterialOutputType.CMOT_FLOAT4)

                    # Adding texture links

                    for i, t in enumerate(texsamples):
                        unreal.MaterialEditingLibrary.connect_material_expressions(t, 'RGBA', x, 't' + str(i))
                    # mat.Expressions = [cust_expr]
                    texcoord = unreal.MaterialEditingLibrary.create_material_expression(mat,
                                                                                        unreal.MaterialExpressionTextureCoordinate,
                                                                                        -3000, -1000)
                    unreal.MaterialEditingLibrary.connect_material_expressions(texcoord, '', x, 'tx')


        unreal.MaterialEditingLibrary.recompile_material(mat)


def add_tex_samples(mat, texs):
    texsamples = []
    for i, tex in enumerate(texs):
        texsample = unreal.MaterialEditingLibrary.create_material_expression(mat,
                                                                             unreal.MaterialExpressionTextureSample,
                                                                             -3000, -700 + 300 * i)
        ts_TextureName = unreal.Paths.get_base_filename(tex + '.png')
        ts_TextureUePath = game_path + texture_path + '/Textures/' + ts_TextureName
        ts_LoadedTexture = unreal.EditorAssetLibrary.load_asset(ts_TextureUePath)
        texsample.set_editor_property('texture', ts_LoadedTexture)
        texsamples.append(texsample)
    return texsamples


def main():
    mats_texs = get_valid_materials_textures()
    print(mats_texs)
    copy_retarget_materials(mats_texs)
    modify_new_materials(mats_texs)


def recompile_materials():
    for asset in unreal.EditorAssetLibrary.list_assets(game_path + material_path):
        asset_data = unreal.EditorAssetLibrary.find_asset_data(asset)
        if asset_data.asset_class != "Material":
            continue
        matname = unreal.Paths.get_base_filename(asset)
        mat = unreal.load_asset(game_path + material_path + matname)
        unreal.MaterialEditingLibrary.recompile_material(mat)

if __name__ == '__main__':
    top_path = 'C:/Users/monta/Documents/Unreal Projects/DynamicShaders/Content/'

    game_path = '/Game/'
    material_path = 'asc_servitor/'
    texture_path = 'asc_servitor/'
    shader_path = 'asc_servitor/'
    estack_template = game_path + '/Template/EStackTemplate'
    done_usfs = []
    main()
    # recompile_materials()

import os
import unreal
# import ast


def get_all_materials():
    for asset in unreal.EditorAssetLibrary.list_assets(game_path + specific_path):
        asset_data = unreal.EditorAssetLibrary.find_asset_data(asset)
        if asset_data.asset_class == "Material":
            modify_material(asset)
    # for file in os.listdir(top_path + specific_path + '/shaders/'):
    #     # Currently only doing base
    #     if '.usf' in file and 'o0' in file and file[:-7] not in done_usfs:
    #         # if '03AB-0C21' in file or '03AB-0C45' in file or '03AB-0EB5' in file:
    #         modify_material(top_path + specific_path + '/shaders/', file)
    #         done_usfs.append([file[:-7]])
            # return


def modify_material(asset):
    usf = asset.split('.')[-1].split('_')[0] + '_o0.usf'
    if usf in os.listdir(top_path + specific_path + '/shaders/'):
        print('usf', usf)
    else:
        print('ERROR ERROR ERROR')
    with open(top_path + specific_path + '/shaders/' + usf) as f:
        f = f.readlines()
        textures = f[1].split("', '")
        textures[0] = textures[0][4:]
        textures[-1] = textures[-1][:-3]
        # textures = ast.literal_eval(f[1][2:])

    matname = unreal.Paths.get_base_filename(asset)
    matpath = game_path + specific_path + matname
    # for asset in unreal.EditorAssetLibrary.list_assets('/Game/'):
    asset_data = unreal.EditorAssetLibrary.find_asset_data(matpath)
    if asset_data.asset_class == "Material":
        mat = unreal.load_asset(matpath)

        texsamples = add_tex_samples(mat, textures)
        custexprs = add_cust_exprs(mat, usf, textures)
        connect_nodes(mat, texsamples, custexprs)
        unreal.MaterialEditingLibrary.recompile_material(mat)
#     file[:-7]


def connect_nodes(mat, texsamples, custexprs):
    # Texcoord and textures to the custom expression nodes
    for custexpr in custexprs:
        for i, t in enumerate(texsamples):
            unreal.MaterialEditingLibrary.connect_material_expressions(t, 'RGBA', custexpr, 't' + str(i))
        # mat.Expressions = [cust_expr]
        texcoord = unreal.MaterialEditingLibrary.create_material_expression(mat,
                                                                            unreal.MaterialExpressionTextureCoordinate,
                                                                            -150, -300)
        unreal.MaterialEditingLibrary.connect_material_expressions(texcoord, '', custexpr, 'tx')

    # Connecting all the final outputs to the properties
    unreal.MaterialEditingLibrary.connect_material_property(custexprs[0], '', unreal.MaterialProperty.MP_BASE_COLOR)
    # unreal.MaterialEditingLibrary.connect_material_property(, '', unreal.MaterialProperty.MP_METALLIC)
    # unreal.MaterialEditingLibrary.connect_material_property(, '', unreal.MaterialProperty.MP_NORMAL)
    # unreal.MaterialEditingLibrary.connect_material_property(, '', unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
    # unreal.MaterialEditingLibrary.connect_material_property(, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    # unreal.MaterialEditingLibrary.connect_material_property(, '', unreal.MaterialProperty.MP_ROUGHNESS)

def add_cust_exprs(material, usf, textures):
    custexprs = []
    for i in range(1):
        custexpr = unreal.MaterialEditingLibrary.create_material_expression(material,
                                                                            unreal.MaterialExpressionCustom,
                                                                            -300, 300*i)
        code = '#include "' + top_path + specific_path + '/shaders/' + usf[:-7] + '_o' + str(i) + '.usf"\nreturn 0;'
        inputs = []
        for i in range(len(textures)):
            ci = unreal.CustomInput()
            ci.set_editor_property('input_name', 't' + str(i))
            inputs.append(ci)
        ci = unreal.CustomInput()
        ci.set_editor_property('input_name', 'tx')
        inputs.append(ci)
        custexpr.set_editor_property('code', code)
        custexpr.set_editor_property('inputs', inputs)
        custexpr.set_editor_property('output_type', unreal.CustomMaterialOutputType.CMOT_FLOAT4)
        custexprs.append(custexpr)
    return custexprs


def add_tex_samples(material, textures):
    texsamples = []
    for i, tex in enumerate(textures):
        texsample = unreal.MaterialEditingLibrary.create_material_expression(material,
                                                                             unreal.MaterialExpressionTextureSample,
                                                                             -600, 300 * i)
        ts_TextureName = unreal.Paths.get_base_filename(tex + '.png')
        ts_TextureUePath = game_path + specific_path + '/Textures/' + ts_TextureName
        ts_LoadedTexture = unreal.EditorAssetLibrary.load_asset(ts_TextureUePath)
        texsample.set_editor_property('texture', ts_LoadedTexture)
        texsamples.append(texsample)
    return texsamples

# add_tex_samples()
top_path = 'C:/Users/monta/Documents/Unreal Projects/MapsShaderTests/Content/'
# top_path = 'C:/Users/monta/Documents/Unreal Projects/ShaderTests/Content/'

game_path = '/Game/'
# specific_path = '/0A49EB80/'
specific_path = 'HangarAndRest/'
done_usfs = []
get_all_materials()

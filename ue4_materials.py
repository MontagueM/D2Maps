import os
import unreal
# import ast


def get_all_materials(path):
    for file in os.listdir(path):
        # Currently only doing base
        if '.usf' in file and 'o0' in file:
            modify_material(path, file)
            # return


def modify_material(path, usf):
    material_name = usf[:-7]
    with open(path + usf) as f:
        f = f.readlines()
        textures = f[1].split("', '")
        textures[0] = textures[0][4:]
        textures[-1] = textures[-1][:-3]
        # textures = ast.literal_eval(f[1][2:])

    print(material_name)
    asset = '/Game/' + material_name
    matname = unreal.Paths.get_base_filename(asset)
    matpath = '/Game/' + matname
    # for asset in unreal.EditorAssetLibrary.list_assets('/Game/'):
    asset_data = unreal.EditorAssetLibrary.find_asset_data(matpath)
    if asset_data.asset_class == "Material":
        mat = unreal.load_asset(matpath)

        texsamples = add_tex_samples(mat, textures)
        cust_expr = add_cust_expr(mat, usf, textures)
        connect_nodes(mat, texsamples, cust_expr)
        unreal.MaterialEditingLibrary.recompile_material(mat)
#     file[:-7]


def connect_nodes(mat, texsamples, cust_expr):
    for i, t in enumerate(texsamples):
        unreal.MaterialEditingLibrary.connect_material_expressions(t, 'RGBA', cust_expr, 't' + str(i))
    # mat.Expressions = [cust_expr]
    texcoord = unreal.MaterialEditingLibrary.create_material_expression(mat,
                                                                        unreal.MaterialExpressionTextureCoordinate,
                                                                        -150, -300)
    unreal.MaterialEditingLibrary.connect_material_expressions(texcoord, '', cust_expr, 'tx')
    unreal.MaterialEditingLibrary.connect_material_property(cust_expr, '', unreal.MaterialProperty.MP_BASE_COLOR)


def add_cust_expr(material, usf, textures):
    custexpr = unreal.MaterialEditingLibrary.create_material_expression(material,
                                                                        unreal.MaterialExpressionCustom,
                                                                        -50, 0)
    code = '#include "C:/Users/monta/Documents/Unreal Projects/MapsShaderTests/Content/shaders/' + usf + '"\nreturn 0;'
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
    return custexpr


def add_tex_samples(material, textures):
    texsamples = []
    for i, tex in enumerate(textures):
        texsample = unreal.MaterialEditingLibrary.create_material_expression(material,
                                                                             unreal.MaterialExpressionTextureSample,
                                                                             -300, 300 * i)
        ts_TextureName = unreal.Paths.get_base_filename(tex + '.png')
        ts_TextureUePath = '/Game/Textures/' + ts_TextureName
        ts_LoadedTexture = unreal.EditorAssetLibrary.load_asset(ts_TextureUePath)
        texsample.set_editor_property('texture', ts_LoadedTexture)
        texsamples.append(texsample)
    return texsamples

# add_tex_samples()
get_all_materials('C:/Users/monta/Documents/Unreal Projects/MapsShaderTests/Content/shaders/')

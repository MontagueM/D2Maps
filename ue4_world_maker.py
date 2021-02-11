"""

Code that automatically imports all obj files from a directory:

    import os
    import unreal

    path = 'C:/d2_maps/orphaned_0932/'

    files = [path + x for x in os.listdir(path) if '.obj' in x][3:]

    for file in files:
        # Create an importing task

        task = unreal.AssetImportTask()
        task.set_editor_property('automated', True)
        task.set_editor_property('destination_name', '')
        task.set_editor_property('destination_path', '/Game/0932/')
        task.set_editor_property('filename', file)
        task.set_editor_property('replace_existing', True)
        task.set_editor_property('save', True)


        # Run the task

        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

Idea to utilise UE4 model instancing:
 - Import all models with zero map-based transformations (can do using the above), but since we have a map
   we can just load the map in not as a blueprint and it just imports all the static meshes separately anyway.
   Could check speed on this, as its more space efficient and sane to keep it as one file.
 - Add copy count number of each model to the level: since I want to be able to run this file dislocated from the
   actual extractor, I should either 1. calculate all the data in here too as its not that slow or 2. store a data file
   that I ask for that stores all the stuff. Static meshes should be named with numeration to make it easier to identify
   which should be next to get the correct transformations
 - Transform each imported asset in the world (Yes eg: unreal.Actor.add_actor_world_transform). The default transform
   should work as the UE4 import is the same as the FBX stuff, but we might have to fiddle a bit with the quat as it
   doesnt like the rotation we give it. If it messes up too much, we can try working with the separate rotate -> translate
   etc since it should go in order and so can work the same way as before the copyefficiency code.

Methods:
 - unreal.WorldFactory
 - unreal.EditorLevelLibrary (better documentation)
"""
import os
import unreal
import json
import time


def import_map(file, target_import_path):
    # The map only needs to consist of materials and all map stuff. There MUST be an index added to these models to help identify what data to use
    task = unreal.AssetImportTask()
    task.set_editor_property('automated', True)
    task.set_editor_property('destination_name', '')
    task.set_editor_property('destination_path', f'/Game/{target_import_path}')
    task.set_editor_property('filename', file)
    task.set_editor_property('replace_existing', True)
    task.set_editor_property('save', True)
    task.options = unreal.FbxSceneImportOptionsStaticMesh()
    task.options.set_editor_property('auto_generate_collision', False)
    task.options.set_editor_property('build_adjacency_buffer', False)
    task.options.set_editor_property('build_reversed_index_buffer', False)
    task.options.set_editor_property('vertex_color_import_option', unreal.FbxSceneVertexColorImportOption.REPLACE)


    # Run the task

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])


# def world_factory():
#     factory = unreal.WorldFactory()
#     # Create new level asset
#     level = factory.factory_create_new('/Game/LevelTest1')
#
#     # Populate with assets
#     for l in loaded_models:
#         s = level.actor_spawn(l)


def editor_level_lib(assets, helper, start, end, level_name):
    # Create new level asset
    unreal.EditorLevelLibrary.new_level(f'/Game/{level_name}')

    # Get transforms
    # transforms = {}
    # for name, tr in helper.items():
    #     t = unreal.EulerTransform(location=tr[0], rotation=scipy.spatial.transform.Rotation.from_quat(tr[1]).as_euler('xyz', degrees=True), scale=[tr[2], tr[2], tr[2]])
    #     transforms[name] = t

    # Populate with assets
    # for i, a in enumerate(assets):
    for i in range(start, end):
        a = assets[i]
        name = a.split('.')[0].split('_unreal_')[-1]
        if '/' in name:
            name = name.split('/')[-1]
        s = name.split('_')
        # start_nums = int(s[4])
        copy_count = int(s[3])
        print(s)
        # name = s[0]
        # name = a.split('.')[0].split('_')[1]

        # print(name)
        # if '0D4BD580' not in name:
        # if '734DD580' not in name and '0D4BD580' not in name:
        #     continue
        # if name != '44A5B580_5_0_0':
        #     continue
        sm = unreal.EditorAssetLibrary.load_asset(a)
        # print(sm)
        # continue
        # print(sm)
        # a = unreal.EditorAssetLibrary.load_asset('/Game/NoCopiesAssets/' + sm)
        # if name not in helper:
        #     print(f'Missing file {name}')
        #     continue
        print(name, copy_count)
        for j in range(copy_count):
            print(copy_count)
            r = helper[name][j][1]
            l = helper[name][j][0]
            l = [-l[0]*100, l[1]*100, l[2]*100]
            rotator = unreal.Rotator(-r[0], r[1], -r[2])
            s = unreal.EditorLevelLibrary.spawn_actor_from_object(sm, location=l, rotation=rotator)  # l must be UE4 Object
            s.set_actor_scale3d([helper[name][j][2]*100]*3)
        if i % 20 == 0:
            unreal.EditorLevelLibrary.save_current_level()


def get_loaded_assets(path):
    return [x for x in unreal.EditorAssetLibrary.list_assets(f'/Game/{path}', recursive=False) if unreal.EditorAssetLibrary.find_asset_data(x).asset_class == 'StaticMesh']


if __name__ == '__main__':
    # Read dictionary

    map_name = 'I:/maps/season_hub_02e4_fbx/02E4-0799_unreal'
    #map_name = 'I:/maps/city_tower_d2_02aa_fbx/02AA-1A83' + '_unreal'

    helper = json.load(open(f'{map_name}.txt'))

    # Test data
    map_path = f'{map_name}.fbx'
    # import_map(map_path, target_import_path='RotTests')
    assets = get_loaded_assets('HELM')
    editor_level_lib(assets, helper, start=int(4*len(assets)/8), end=int(8*len(assets)/8), level_name='HELM')

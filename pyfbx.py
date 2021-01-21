import fbx
import sys


class Node:
    def __init__(self, parent, name='', children=None):
        self._me = fbx.FbxNode.Create(parent._me, name)
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def add_child(self, node):
        assert isinstance(node, Node)
        self.children.append(node)

    def add_attribute(self, attr):
        self.attribute = attr
        self.attribute._me = attr._me
        return self._me.AddNodeAttribute(attr._me)


class Manager:
    nodes = []

    def __init__(self):
        self._me = fbx.FbxManager.Create()

    def create_scene(self, name=""):
        self.scene = Scene(self, name)

    def create_node(self, name=""):
        node = Node(self, name)
        self.nodes.append(node)

    def get_nodes(self):
        return self.nodes


class Scene:
    def __init__(self, manager, name=""):
        self._me = fbx.FbxScene.Create(manager._me, name)
        self.manager = manager
        self.name = name

        self.root_node = Node(self, name + "_root_node")
        self.root_node._me = self._me.GetRootNode()


class Mesh:
    def __init__(self, parent, name=""):
        self.parent = parent
        self.name = name
        self._me = fbx.FbxMesh.Create(parent._me, name)

    def add_to(self, node):
        node.add_attribute(self)


class Model:
    manager = None
    importer = None
    exporter = None
    scene = None
    root_node = None
    cpt_count = 0

    def __init__(self):
        """Create an atomic manager, exporter, scene and its root node."""
        Model.manager = fbx.FbxManager.Create()
        if not Model.manager:
            sys.exit(0)

        Model.ios = fbx.FbxIOSettings.Create(Model.manager, fbx.IOSROOT)
        Model.importer = fbx.FbxImporter.Create(Model.manager, '')
        Model.exporter = fbx.FbxExporter.Create(Model.manager, '')
        Model.scene = fbx.FbxScene.Create(Model.manager, '')
        Model.root_node = Model.scene.GetRootNode()

    def import_file(self, import_path=None, ascii_format=False):
        if not Model.manager.GetIOSettings():
            Model.ios = fbx.FbxIOSettings.Create(Model.manager, fbx.IOSROOT)
            Model.manager.SetIOSettings(Model.ios)
        if ascii_format:
            b_ascii = 1
        else:
            b_ascii = -1
        importstat = Model.importer.Initialize(import_path, b_ascii, Model.manager.GetIOSettings())

        if not importstat:
            try:
                raise IOError("Problem importing file!")
            except IOError as e:
                print("An exception flew by!")

        importstat = Model.importer.Import(Model.scene)

        Model.importer.Destroy()

    def export(self, save_path=None, ascii_format=False):
        """Export the scene to an fbx file."""

        if not Model.manager.GetIOSettings():
            Model.ios = fbx.FbxIOSettings.Create(Model.manager, fbx.IOSROOT)
            Model.manager.SetIOSettings(Model.ios)

        Model.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
        Model.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
        Model.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_EMBEDDED, False)
        Model.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_SHAPE, True)
        Model.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GOBO, False)
        Model.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_ANIMATION, False)
        Model.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GLOBAL_SETTINGS, True)
        if ascii_format:
            b_ascii = 1
        else:
            b_ascii = -1

        exportstat = Model.exporter.Initialize(save_path, b_ascii, Model.manager.GetIOSettings())

        if not exportstat:
            try:
                raise IOError("Problem exporting file!")
            except IOError as e:
                print("An exception flew by!")

        exportstat = Model.exporter.Export(Model.scene)

        Model.exporter.Destroy()

        return exportstat

    def create_node(self, nickname=''):
        """Create a free node and add to the root node."""
        self.node = fbx.FbxNode.Create(Model.manager, nickname)
        Model.root_node.AddChild(self.node)

    def create_mesh(self, nickname=''):
        """Create a free mesh"""
        self.mesh = fbx.FbxMesh.Create(Model.manager, nickname)

    def create_mesh_controlpoint(self, x, y, z):
        """Create a mesh controlpoint."""
        self.mesh.SetControlPointAt(fbx.FbxVector4(x, y, z), Model.cpt_count)
        Model.cpt_count += 1

    def set_rnode_translation(self, coordinates):
        """Set the root node translation."""
        Model.root_node.LclTranslation.Set(fbx.FbxDouble3(
            float(coordinates[0]),
            float(coordinates[1]),
            float(coordinates[2])
        ))

    def set_node_translation(self, coordinates):
        """Set the child node translation."""
        self.node.LclTranslation.Set(fbx.FbxDouble3(
            float(coordinates[0]),
            float(coordinates[1]),
            float(coordinates[2])
        ))

    def set_node_rotation(self, coordinates):
        """Set the child node translation."""
        self.node.LclRotation.Set(fbx.FbxDouble3(
            float(coordinates[0]),
            float(coordinates[1]),
            float(coordinates[2])
        ))

    def set_mesh_to_node(self):
        """Set the mesh to the child node's attribute."""
        self.node.AddNodeAttribute(self.mesh)

    def destroy(self):
        """Free the manager's memory."""
        Model.manager.Destroy()

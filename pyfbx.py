from __future__ import print_function

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

    def create_node(self, name=""):
        node = Node(self, name)


class Mesh:
    def __init__(self, parent, name=""):
        self.parent = parent
        self.name = name
        self._me = fbx.FbxMesh.Create(parent._me, name)

    def add_to(self, node):
        node.add_attribute(self)


class FBox:
    """Wrap fbx's common Python classes"""
    manager = None
    importer = None
    exporter = None
    scene = None
    root_node = None
    cpt_count = 0

    def __init__(self):
        """Create an atomic manager, exporter, scene and its root node."""
        FBox.manager = fbx.FbxManager.Create()
        if not FBox.manager:
            sys.exit(0)

        FBox.ios = fbx.FbxIOSettings.Create(FBox.manager, fbx.IOSROOT)
        FBox.importer = fbx.FbxImporter.Create(FBox.manager, '')
        FBox.exporter = fbx.FbxExporter.Create(FBox.manager, '')
        FBox.scene = fbx.FbxScene.Create(FBox.manager, '')
        FBox.root_node = FBox.scene.GetRootNode()

    def import_file(self, import_path=None, ascii_format=False):
        if not FBox.manager.GetIOSettings():
            FBox.ios = fbx.FbxIOSettings.Create(FBox.manager, fbx.IOSROOT)
            FBox.manager.SetIOSettings(FBox.ios)
        if ascii_format:
            b_ascii = 1
        else:
            b_ascii = -1
        importstat = FBox.importer.Initialize(import_path, b_ascii, FBox.manager.GetIOSettings())

        if not importstat:
            try:
                raise IOError("Problem importing file!")
            except IOError as e:
                print("An exception flew by!")

        importstat = FBox.importer.Import(FBox.scene)

        FBox.importer.Destroy()

    def export(self, save_path=None, ascii_format=False):
        """Export the scene to an fbx file."""

        if not FBox.manager.GetIOSettings():
            FBox.ios = fbx.FbxIOSettings.Create(FBox.manager, fbx.IOSROOT)
            FBox.manager.SetIOSettings(FBox.ios)

        FBox.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
        FBox.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
        FBox.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
        FBox.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_SHAPE, True)
        FBox.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GOBO, False)
        FBox.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_ANIMATION, False)
        FBox.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GLOBAL_SETTINGS, True)
        if ascii_format:
            b_ascii = 1
        else:
            b_ascii = -1
        exportstat = FBox.exporter.Initialize(save_path, b_ascii, FBox.manager.GetIOSettings())

        if not exportstat:
            try:
                raise IOError("Problem exporting file!")
            except IOError as e:
                print("An exception flew by!")

        exportstat = FBox.exporter.Export(FBox.scene)

        FBox.exporter.Destroy()

        return exportstat

    def create_node(self, nickname=''):
        """Create a free node and add to the root node."""
        self.node = fbx.FbxNode.Create(FBox.manager, nickname)
        FBox.root_node.AddChild(self.node)

    def create_mesh(self, nickname=''):
        """Create a free mesh"""
        self.mesh = fbx.FbxMesh.Create(FBox.manager, nickname)

    def create_mesh_controlpoint(self, x, y, z):
        """Create a mesh controlpoint."""
        self.mesh.SetControlPointAt(fbx.FbxVector4(x, y, z), FBox.cpt_count)
        FBox.cpt_count += 1

    def get_rnode_translation(self):
        """Get the root node's translation."""
        return FBox.root_node.LclTranslation.Get()

    def get_node_translation(self):
        """Get the child node's translation."""
        return self.node.LclTranslation.Get()

    def set_rnode_translation(self, coordinates):
        """Set the root node translation."""
        FBox.root_node.LclTranslation.Set(fbx.FbxDouble3(
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

    def set_mesh_to_node(self):
        """Set the mesh to the child node's attribute."""
        self.node.AddNodeAttribute(self.mesh)

    def destroy(self):
        """Free the manager's memory."""
        FBox.manager.Destroy()


class PyramidMarker:

    def __init__(self, scene, name):
        self.scene = scene
        self.name = name

    def create(self, base_width, height):
        self.base_width = base_width
        self.height = height

        pyramid = fbx.FbxMesh.Create(self.scene, self.name)

        # Calculate the vertices of the pyramid lying down
        base_width_half = base_width / 2
        controlpoints = [
            fbx.FbxVector4(0, height, 0),
            fbx.FbxVector4(base_width_half, 0, base_width_half),
            fbx.FbxVector4(base_width_half, 0, -base_width_half),
            fbx.FbxVector4(-base_width_half, 0, -base_width_half),
            fbx.FbxVector4(-base_width_half, 0, base_width_half)
        ]

        # Initialize and set the control points of the mesh
        controlpoint_count = len(controlpoints)
        pyramid.InitControlPoints(controlpoint_count)
        for i, p in enumerate(controlpoints):
            pyramid.SetControlPointAt(p, i)

        # Set the control point indices of the bottom plane of the pyramid
        pyramid.BeginPolygon()
        pyramid.AddPolygon(1)
        pyramid.AddPolygon(4)
        pyramid.AddPolygon(3)
        pyramid.AddPolygon(2)
        pyramid.EndPolygon()

        # Set the control point indices of the front plane of the pyramid
        pyramid.BeginPolygon()
        pyramid.AddPolygon(0)
        pyramid.AddPolygon(1)
        pyramid.AddPolygon(2)
        pyramid.EndPolygon()

        # Set the control point indices of the left plane of the pyramid
        pyramid.BeginPolygon()
        pyramid.AddPolygon(0)
        pyramid.AddPolygon(2)
        pyramid.AddPolygon(3)
        pyramid.EndPolygon()

        # Set the control point indices of the back plane of the pyramid
        pyramid.BeginPolygon()
        pyramid.AddPolygon(0)
        pyramid.AddPolygon(3)
        pyramid.AddPolygon(4)
        pyramid.EndPolygon()

        # Set the control point indices of the right plane of the pyramid
        pyramid.BeginPolygon()
        pyramid.AddPolygon(0)
        pyramid.AddPolygon(4)
        pyramid.AddPolygon(1)
        pyramid.EndPolygon()

        # Attach the mesh to a node
        pyramid_node = fbx.FbxNode.Create(self.scene, '')
        pyramid_node.SetNodeAttribute(pyramid)

        self.pyramid_node = pyramid_node

        return pyramid_node

    def set_local_translation(self, coordinate):
        # TODO: Set the world coordinate to Z-up instead of Y-up
        x = float(coordinate[0])
        y = float(coordinate[1])
        z = float(coordinate[2])
        self.pyramid_node.LclTranslation.Set(fbx.FbxDouble3(x, z, y))

    def attach_to_rootnode(self):
        self.scene.GetRootNode().AddChild(self.pyramid_node)

    def set_rotation_pivot(self, coordinate):
        self.pyramid_node.SetRotationActive(True)
        x = float(coordinate[0])
        y = float(coordinate[1])
        z = float(coordinate[2])
        self.pyramid_node.SetRotationPivot(fbx.FbxNode.eSourcePivot, fbx.FbxVector4(x, y, z))

    def set_post_rotation(self, coordinate):
        x = float(coordinate[0])
        y = float(coordinate[1])
        z = float(coordinate[2])
        self.pyramid_node.SetPostRotation(fbx.FbxNode.eSourcePivot, fbx.FbxVector4(x, y, z))


def main(arg, args):
    # args = tuple(sys.argv[2:])

    fb = FBox()
    fb.create_node()
    fb.create_mesh()
    fb.create_mesh_controlpoint(0, 0, 0)

    marker = PyramidMarker(fb.scene, "marker")
    marker.create(4, 4)
    marker.attach_to_rootnode()

    if arg == 'test':
        print("TEST MODE: Creating a vertex at provided x, y, z cartesian coordinates.")
        fb.set_rnode_translation((0, 0, 0,))
        # fb.set_node_translation(args)
        marker.set_local_translation(args)

    elif arg == 'geo':
        print("GEO MODE: Creating a vertex at provided lat, lon, ele geolocation.\n")
        print(
            "Remember in origin point in FBX will represent the origin of the respective UTM zone the location is at. See `https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system` for info.\n")
        geo = ProjGeo(args)
        geo_args = geo.get_projected_coordinates()
        print("Your zone is UTM {}".format(geo.get_utm_zone()))
        print("Your zone's origin real lat, lon is: ", geo.get_utm_zone_origin())
        fb.set_rnode_translation((0, 0, 0,))
        # fb.set_node_translation(geo_args)
        marker.set_local_translation(geo_args)
        print("Marker is set at the projected coordinate: ({0}, {1}, {2})".format(
            geo_args[0], geo_args[1], geo_args[2]))

    else:
        raise ValueError("Please provide `test` or `geo` as first parameter")

    marker.set_rotation_pivot((0, marker.height / 2, 0))
    marker.set_post_rotation((0, 0, 180))

    print("Exporting file...\n")
    fb.export()
    fb.destroy()
    print("Check marker.fbx in this current directory.\n")


if __name__ == '__main__':
    main('test', (0, 0, 0))
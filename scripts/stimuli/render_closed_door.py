""" BPY script that handles rendering logic for scenes
"""
import os
import sys
try:
    import bpy
    import mathutils
except:
    # For documentation
    print('No `bpy` available')
import json
import time
import argparse

# Flush stdout in case blender is complaining
sys.stdout.flush()

class Scene:

    """
    Defines the ramp world in bpy.
    """

    def __init__(self, scene, flip = False):
        """ Initializes objects, physics, and camera

        :param scene: Describes the ramp, table, and balls.
        :type scene_d: dict
        :param trace: the physical state of the objects
        :type trace: dict or None
        :param theta: Angle around the world to point the camera
        :type theta: float or None
        """
        if flip:
            bpy.data.objects["painting"].select_set(True)
            # to select the object in the 3D viewport,
            current_state = bpy.data.objects["painting"]
            current_state.location[0] *= -1
            # for obj in bpy.data.objects:
            #     if obj.name != 'Camera':
            #         obj.location[0] *= -1

        bpy.context.view_layer.update()

        # Parse scene structure
        self.load_scene(scene)
        print('Loaded scene')
        sys.stdout.flush()

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, t):
        """
        A dictionary containing to keys: `('pos', 'rot')` where each
        holds a 3-dimensional array of `TxNxK`,
        where `T` is the number of key frames,
        `N` is the number of objects,
        and `K` is either `xyz` or `wxyz`.

        :param t: The physics state to apply as keyframes
        :type t: dict or None
        """
        if not t is None:
            frames = len(t['pos'])
        else:
            frames = 1
        bpy.context.scene.frame_set(1)
        bpy.context.scene.frame_end = frames + 1
        self._trace = t

    def select_obj(self, obj):
        """ Sets the given object into active context.
        """
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.update()


    def rotate_obj(self, obj, rot):
        """ Rotates the object.

        :param rot: Either an euler angle (xyz) or quaternion (wxyz)
        """
        self.select_obj(obj)
        if len(rot) == 3:
            obj.rotation_mode = 'XYZ'
            obj.rotation_euler = rot
        else:
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = np.roll(rot, 1) # [3, 0, 1, 2]
        bpy.context.view_layer.update()

    def move_obj(self, obj, pos):
        """ Moves the object.

        :param pos: An xyz designating the object's new location.
        """
        self.select_obj(obj)
        pos = mathutils.Vector(pos)
        obj.location = pos
        bpy.context.view_layer.update()

    def scale_obj(self, obj, dims):
        """ Rescales to the object to the given dimensions.
        """
        self.select_obj(obj)
        obj.dimensions = dims
        bpy.context.view_layer.update()
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        bpy.context.view_layer.update()

    def set_appearance(self, obj, mat):
        """ Assigns a material to a block.
        """
        if mat in bpy.data.materials:
            obj.active_material = bpy.data.materials[mat]
        bpy.context.view_layer.update()

    def create_obj(self, name, object_d):
        """ Initializes a block.

        :param name: The name to refer to the object
        :type name: str
        :param object_d: Describes the objects appearance and location.
        :type object_d: dict
        """
        if object_d['shape'] == 'Ball':
            bpy.ops.mesh.primitive_ico_sphere_add(location=object_d['position'],
                                                  enter_editmode=False,
                                                  subdivisions=7,
                                                  radius = 1)
            ob = bpy.context.object
            self.scale_obj(ob, object_d['dims'])
        elif object_d['shape'] == 'Block':
            bpy.ops.mesh.primitive_cube_add(location=object_d['position'],
                                            enter_editmode=False,)
            ob = bpy.context.object
            self.scale_obj(ob, object_d['dims'])
            self.rotate_obj(ob, object_d['orientation'])
        elif object_d['shape'] == 'Puck':
            bpy.ops.mesh.primitive_cylinder_add(
                location=object_d['position'],
                enter_editmode=False,)
            ob = bpy.context.object
            self.scale_obj(ob, object_d['dims'])
            self.rotate_obj(ob, object_d['orientation'])
        elif object_d['shape'] == 'Plane':
            bpy.ops.mesh.primitive_plane_add(
                location=object_d['position'],
                enter_editmode=False,)
            ob = bpy.context.object
            self.scale_obj(ob, object_d['dims'])
            self.rotate_obj(ob, object_d['orientation'])
        else:
            raise ValueError('Not supported')


        ob = bpy.context.object
        ob.name = name
        ob.show_name = True
        me = ob.data
        me.name = '{0!s}_Mesh'.format(name)

        if 'appearance' in object_d:
            mat = object_d['appearance']
            # HACK: this is a block
            if mat == "blue":
                ob.data.materials.append(mat_blue2)
                ob.data.materials.append(bpy.data.materials[mat])
                ob.data.polygons[5].material_index = 1
        else:
            mat = 'U'
            self.set_appearance(ob, mat)

    def load_scene(self, scene_dict):
        """ Configures the ramp, table, and balls
        """
        # Load Objects / Tiles
        obj_data = scene_dict['objects']
        obj_names = list(map(str, range(len(obj_data))))
        self.obj_names = obj_names
        for i in range(len(obj_data)):
            name = obj_names[i]
            data = obj_data[i]
            mat = data['appearance']
            # only create obstacles
            if mat == "blue":
                self.create_obj(name, data)

    def set_rendering_params(self, resolution):
        """ Configures various settings for rendering such as resolution.
        """
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.engine = 'CYCLES'
        # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        # bpy.context.scene.cycles.samples = 128
        # bpy.context.scene.render.tile_x = 16
        # bpy.context.scene.render.tile_y = 16

    def set_camera(self, params):
        """ Moves the camera along a circular path.

        :param rot: Angle in radians along path.
        :type rot: float
        """
        camera = bpy.data.objects['Camera']
        camera.location = params['position']
        self.rotate_obj(camera, params['orientation'])
        bpy.context.view_layer.update()
        camera.keyframe_insert(data_path='location', index = -1)
        camera.keyframe_insert(data_path='rotation_quaternion', index = -1)

    def set_lights(self, lights):
        for (i, l) in enumerate(lights):

            # cribbed from https://stackoverflow.com/a/57310198
            name = 'light_{0:d}'.format(i)
            # create light datablock, set attributes
            light_data = bpy.data.lights.new(name=name, type='AREA')
            light_data.energy = l['intensity']

            # create new object with our light datablock
            light_object = bpy.data.objects.new(name=name, object_data=light_data)

            # link light object
            bpy.context.collection.objects.link(light_object)

            # make it active
            bpy.context.view_layer.objects.active = light_object

            #change location
            light_object.location = l['position']
            self.rotate_obj(light_object, l['orientation'])
            bpy.context.view_layer.update()


    def render(self, output_name, resolution , camera_rot = None):
        """ Renders a scene.

        Skips over existing frames

        :param output_name: Path to save frames
        :type output_name: str
        :param frames: a list of frames to render (shifted by warmup)
        :type frames: list
        :param resolution: Image resolution
        :type resolution: tuple(int, int)
        :param camera_rot: Rotation for camera.
        :type camera_rot: float

        """
        if not (resolution is None):
            self.set_rendering_params(resolution)

        if os.path.isfile(output_name):
            print('File {0!s} exists'.format(output_name))
            return

        bpy.context.scene.render.filepath = output_name
        t0 = time.time()
        print('Rendering ')
        sys.stdout.flush()
        with Suppressor():
            bpy.ops.render.render(write_still=True)
        print('Rendering took {}s'.format(time.time() - t0))
        sys.stdout.flush()


    def save(self, out):
        """
        Writes the scene as a blend file.
        """
        bpy.ops.wm.save_as_mainfile(filepath=out)

# From https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class Suppressor(object):

    # A context manager for doing a "deep suppression" of stdout and stderr in
    # Python, i.e. will suppress all print, even if the print originates in a
    # compiled C/Fortran sub-function.

    # This will not suppress raised exceptions, since exceptions are printed
    # to stderr just before a script exits, and after the context manager has
    # exited (at least, I think that is why it lets exceptions through).

    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

# from https://blender.stackexchange.com/a/240875
def create_material(mat_name, diffuse_color=(1,1,1,1)):
    mat = bpy.data.materials.new(name=mat_name)
    mat.diffuse_color = diffuse_color
    return mat

# Generate 2 demo materials
mat_red = create_material("Red", (1,0,0,1))
mat_blue2 = create_material("Blue2", (0.01,0.4,0.9,1.0))

def parser(args):
    """Parses extra arguments
    """
    p = argparse.ArgumentParser(description = 'Renders blockworld scene')
    p.add_argument('--scene', type = load_data,
                   help = 'json serialized string describing the scene.')
    p.add_argument('--out', type = str,
                   help = 'Path to save rendering')
    p.add_argument('--mode', type = str, default = 'noflip',
                   choices = ['noflip', 'flip'],
                   help = 'mode to render')
    p.add_argument('--resolution', type = int, nargs = 2,
                   help = 'Render resolution')
    return p.parse_args(args)

def load_data(path):
    """Helper that loads trace file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def main():
    argv = sys.argv
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser(argv)


    path = os.path.join(args.out, 'render')
    if not os.path.isdir(path):
        os.mkdir(path)


    scene = Scene(args.scene, flip = args.mode == 'flip')

    p = args.out + '.png'
    scene.render(p, resolution = args.resolution,)
    # if args.save_world:
    path = os.path.join(args.out, 'world.blend')
    scene.save(path)

if __name__ == '__main__':
    main()
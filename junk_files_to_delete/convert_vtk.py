import vtk
import glob

if __name__ == '__main__':
    # convert vtp to ply
    mesh_dir = "/Geometry_left"
    mesh_file = glob.glob(mesh_dir + "/*.vtp")
    if len(mesh_file) == 0:
        raise FileNotFoundError("No mesh file found")
    for file in mesh_file:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file)
        reader.Update()
        # invert normals
        normal = vtk.vtkPolyDataNormals()
        normal.SetInputConnection(reader.GetOutputPort())
        normal.FlipNormalsOn()
        normal.Update()
        reader = normal

        writer = vtk.vtkPLYWriter()
        writer.SetFileName(file[:-4] + ".ply")
        writer.SetInputConnection(reader.GetOutputPort())
        writer.Write()
        print("Conversion done")
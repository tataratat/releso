import os
import pathlib
import shutil

import pytest
from pydantic import ValidationError

from releso.mesh import MeshExporter, MeshIOMesh, MixdMesh

# will look into gustaf test meshes to test these classes


@pytest.mark.parametrize(
    "path, mesh_format, expected, error",
    [
        ("test", "test", False, "mesh_format='test' is not supported"),
        (
            "test.txt",
            "mixd",
            False,
            "The format and the path suffix do not match. The format",
        ),
        ("test.xns", "mixd", pathlib.Path("test.xns").resolve(), None),
        ("test", "mixd", pathlib.Path("_.xns").resolve(), None),
        ("test/test", "mixd", pathlib.Path("test/_.xns").resolve(), None),
    ],
)
def test_mesh_exporter_export_path_format(
    path, mesh_format, expected, error, dir_save_location
):
    calling_dict = {
        "export_path": path,
        "format": mesh_format,
        "save_location": dir_save_location,
    }
    if error:
        with pytest.raises(ValidationError) as err:
            MeshExporter(**calling_dict)
        assert error in str(err.value)
        return
    mesh_exporter = MeshExporter(**calling_dict)
    assert mesh_exporter.export_path == expected
    temp_expected = pathlib.Path(str(expected).format(0))
    assert mesh_exporter.get_export_path() == temp_expected
    env_id = "123"
    mesh_exporter.adapt_export_path(env_id)
    expected = pathlib.Path(str(expected).format(env_id))
    assert mesh_exporter.get_export_path() == expected
    if "/" in path:
        os.rmdir(expected.parent)


@pytest.mark.parametrize(
    "load_sample_file",
    [
        "volumes/tet/3DBrickTet.msh",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    (
        "path",
        "mxyz_path",
        "mien_path",
        "hyper_cube",
        "export",
        "dimensions",
        "w_level",
        "warning",
        "error",
    ),
    [
        (
            ".xns",
            None,
            None,
            False,
            {"format": "mixd", "export_path": "test"},
            3,
            False,
            False,
            False,
        ),
        (
            ".xns",
            None,
            None,
            False,
            None,
            3,
            "export",
            "No mesh exporter definition. Please define one",
            False,
        ),
        (
            "mxyz",
            None,
            None,
            False,
            {"format": "mixd", "export_path": "test"},
            3,
            False,
            False,
            False,
        ),
        (
            "mien",
            None,
            None,
            False,
            {"format": "mixd", "export_path": "test"},
            3,
            False,
            False,
            False,
        ),
        (
            "mxyz_no_name",
            None,
            None,
            False,
            {"format": "mixd", "export_path": "test"},
            3,
            False,
            False,
            False,
        ),
        (
            "mien_no_name",
            None,
            None,
            False,
            {"format": "mixd", "export_path": "test"},
            3,
            False,
            False,
            False,
        ),
        (
            None,
            "correct",
            "correct",
            False,
            {"format": "mixd", "export_path": "test"},
            3,
            False,
            False,
            False,
        ),
        (
            "msh",
            None,
            None,
            False,
            {"format": "mixd", "export_path": "test"},
            3,
            False,
            False,
            "Could not locate mien nor mxyz file path.",
        ),
        (
            None,
            None,
            None,
            False,
            {"format": "mixd", "export_path": "test"},
            3,
            False,
            False,
            "Could not locate the correct mien/mxyz paths.",
        ),
    ],
)
def test_mixd_mesh_class(
    path,
    mxyz_path,
    mien_path,
    hyper_cube,
    export,
    dimensions,
    w_level,
    warning,
    error,
    load_sample_file,
    dir_save_location,
    caplog,
):
    # setup mixd
    if not load_sample_file.with_suffix(".mien").exists():
        from gustaf.io import load
        from gustaf.io.mixd import export as export_io

        mesh = load(load_sample_file)[-1]
        export_io(load_sample_file.with_suffix(".xns"), mesh)
        shutil.copy(
            load_sample_file.with_suffix(".mxyz"),
            load_sample_file.parent / "mxyz",
        )
        shutil.copy(
            load_sample_file.with_suffix(".mxyz"),
            load_sample_file.parent / "mxyz.space",
        )
        shutil.copy(
            load_sample_file.with_suffix(".mien"),
            load_sample_file.parent / "mien",
        )
    load_sample_file = load_sample_file.with_suffix(".xns")
    calling_dict = {
        "save_location": dir_save_location,
    }
    if path:
        if path == ".xns":
            path = load_sample_file.parent
        elif path == "mxyz":
            path = load_sample_file.with_suffix(".mxyz")
        elif path == "mien":
            path = load_sample_file.with_suffix(".mien")
        elif path == "mxyz_no_name":
            path = load_sample_file.with_name("mxyz")
        elif path == "mien_no_name":
            path = load_sample_file.with_name("mien")
        elif path == "msh":
            path = load_sample_file.with_suffix(".msh")
        calling_dict["path"] = path
    if mxyz_path:
        if mxyz_path == "correct":
            mxyz_path = load_sample_file.with_suffix(".mxyz")
        calling_dict["mxyz_path"] = mxyz_path
    if mien_path:
        if mien_path == "correct":
            mien_path = load_sample_file.with_suffix(".mien")
        calling_dict["mien_path"] = mien_path
    if hyper_cube:
        calling_dict["hyper_cube"] = hyper_cube
    if dimensions:
        calling_dict["dimensions"] = dimensions
    if export:
        export["save_location"] = dir_save_location
        calling_dict["export"] = MeshExporter(**export)
    if error:
        with pytest.raises(ValidationError) as err:
            MixdMesh(**calling_dict)
        assert error in str(err.value)
        return
    mesh = MixdMesh(**calling_dict)
    if w_level == "export":
        assert mesh.get_export_path() is None
        with caplog.at_level("WARNING"):
            assert mesh.adapt_export_path("123") is None
            assert warning in caplog.text
    else:
        assert mesh.get_export_path() is not None


@pytest.mark.parametrize(
    "load_sample_file, dimension, change_path, cp_rm, error",
    [
        ("volumes/tet/3DBrickTet.msh", 3, False, False, False),
        # currently not supported
        # ("volumes/tet/3DBrickTet.msh", ".msh2", True, False),
        (
            "volumes/tet/3DBrickTet.msh",
            3,
            ".msh2",
            False,
            "Could not locate the mesh file.",
        ),
        (
            "volumes/tet/3DBrickTet.msh",
            3,
            ".msh3",
            True,
            "Mesh type not supported.",
        ),
        (
            "volumes/tet/3DBrickTet.msh",
            3,
            "rm",
            False,
            "For MeshIO base mesh import a path must be provided.",
        ),
        ("faces/quad/2DChannelQuad.msh", 2, False, False, False),
    ],
    indirect=["load_sample_file"],
)
def test_meshio_mesh(
    load_sample_file, dimension, change_path, cp_rm, error, dir_save_location
):
    if change_path:
        if change_path == "rm":
            load_sample_file = None
        else:
            load_sample_file = load_sample_file.with_suffix(change_path)
            if cp_rm:
                shutil.copy(
                    load_sample_file.with_suffix(".msh"),
                    load_sample_file,
                )
    call_dict = {
        "path": load_sample_file,
        "dimensions": dimension,
        "save_location": dir_save_location,
    }
    if error:
        with pytest.raises(ValidationError) as err:
            MeshIOMesh(**call_dict)
        assert error in str(err.value)
        if change_path and cp_rm:
            os.remove(load_sample_file)
        return
    mesh = MeshIOMesh(**call_dict)
    if change_path and cp_rm:
        os.remove(load_sample_file)
    mesh.get_mesh()


@pytest.mark.parametrize(
    "load_sample_file", ["faces/quad/2DChannelQuad.msh"], indirect=True
)
def test_mixd_mesh_export_and_get(
    load_sample_file, dir_save_location, clean_up_provider, caplog
):
    mesh_exporter = MeshExporter(
        format="mixd",
        export_path=f"{str(dir_save_location)}/export/" + "{}/test.xns",
        save_location=dir_save_location,
    )
    mesh = MeshIOMesh(
        path=load_sample_file,
        dimensions=2,
        save_location=dir_save_location,
        export=mesh_exporter,
    )
    mesh.adapt_export_path("123")
    mesh.export.export_mesh(mesh.get_mesh())
    exported_file = pathlib.Path(f"{str(dir_save_location)}/export")
    mesh_exporter.mesh_format = "other"
    with pytest.raises(RuntimeError) as err:
        mesh_exporter.export_mesh(None)
    assert "The requested format other is not supported." in str(err.value)

    mixd_mesh = MixdMesh(
        path="export/123/test.xns",
        save_location=dir_save_location,
        dimensions=2,
    )
    with caplog.at_level("INFO"):
        mixd_mesh.get_mesh()

    clean_up_provider(exported_file)

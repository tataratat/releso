"""
Author: Daniel Wolf (wolf@avt.rwth-aachen.de)
"""
import gustav as gus
import vedo
import pathlib



def plot_undeformed_unit_mesh(FFD: gus.FreeFormDeformation, path: pathlib.Path):

    undeformed_unit_mesh_axes = dict(
        # general stuff
        titleFont='SmartCouric',
        labelFont="LogoType",
        axesLineWidth= 4,
        gridLineWidth= 3,
        xyPlaneColor='gray',
        xyGridColor='k1',
        # x axis details
        xtitle='x [-]', # latex-style syntax
        xrange=[0, 1.05],
        xValuesAndLabels=[(0,'0.0'),(0.2,'0.2'),(0.4,'0.4'),(0.6,'0.6'),
            (0.8,'0.8'), (1.0,'1.0')],
        xTitlePosition=0.5,  # title fractional positions along axis
        xTitleJustify='top-center', # align title wrt to its axis
        xTitleSize=0.035,
        xTitleOffset=0.05,
        xLabelSize=0.025,
        # y axis details
        ytitle='y [-]',
        yrange=[0, 1.05],
        yValuesAndLabels=[(0,'0.0'),(0.2,'0.2'),(0.4,'0.4'),(0.6,'0.6'),
            (0.8,'0.8'), (1.0,'1.0')],
        yTitlePosition=0.5,  # title fractional positions along axis
        yTitleJustify='top-center', # align title wrt to its axis
        yTitleSize=0.035,
        yTitleOffset=0.12,
        yLabelSize=0.025,    # size of the numeric labels along Y axis
        # background
        xyFrameColor='k1'
    )

    # undeformed_unit_mesh
    undeformed_unit_mesh = FFD.unit_mesh_
    vedo_undeformed_unit_mesh = undeformed_unit_mesh.vedo_mesh
    vedo_undeformed_unit_mesh.wireframe(True)
    vedo_undeformed_unit_mesh.lineWidth(3)
    plot = vedo.show(vedo_undeformed_unit_mesh, axes=undeformed_unit_mesh_axes, 
        offscreen=True, size=(4800,4800))
    #plot = vedo.show(vedo_undeformed_unit_mesh, axes=undeformed_unit_mesh_axes, 
    #    size=(4800,4800),interactive=True).close()
    plot.screenshot(path)

# plot_undeformed_unit_mesh

def plot_deformed_unit_mesh(FFD: gus.FreeFormDeformation, path: pathlib.Path):

    deformed_unit_mesh_axes = dict(
        # general stuff
        titleFont='SmartCouric',
        labelFont="LogoType",
        axesLineWidth= 4,
        gridLineWidth= 3,
        xyPlaneColor='gray',
        xyGridColor='k1',
        # x axis details
        xtitle='x [-]', # latex-style syntax
        xrange=[0, 1.05],
        xValuesAndLabels=[(0,'0.0'),(0.2,'0.2'),(0.4,'0.4'),(0.6,'0.6'),
            (0.8,'0.8'), (1.0,'1.0')],
        xTitlePosition=0.5,  # title fractional positions along axis
        xTitleJustify='top-center', # align title wrt to its axis
        xTitleSize=0.035,
        xTitleOffset=0.05,
        xLabelSize=0.025,
        # y axis details
        ytitle='y [-]',
        yrange=[0, 1.15],
        yValuesAndLabels=[(0,'0.0'),(0.2,'0.2'),(0.4,'0.4'),(0.6,'0.6'),
            (0.8,'0.8'), (1.0,'1.0')],
        yTitlePosition=0.5,  # title fractional positions along axis
        yTitleJustify='top-center', # align title wrt to its axis
        yTitleSize=0.035,
        yTitleOffset=0.12,
        yLabelSize=0.025,    # size of the numeric labels along Y axis
        # background
        xyFrameColor='k1'
    )

    # deformed unit mesh
    deformed_unit_mesh = FFD.deformed_unit_mesh_
    vedo_deformed_unit_mesh = deformed_unit_mesh.vedo_mesh
    vedo_deformed_unit_mesh.wireframe(True)
    vedo_deformed_unit_mesh.lineWidth(3)
    plot = vedo.show(vedo_deformed_unit_mesh, axes=deformed_unit_mesh_axes, 
        offscreen=True, size=(4800,4800))
    #plot = vedo.show(vedo_deformed_unit_mesh, axes=deformed_unit_mesh_axes, 
    #    size=(4800,4800), interactive=True).close()
    plot.screenshot(path)

# plot_deformed_unit_mesh

def plot_undeformed_mesh(FFD: gus.FreeFormDeformation, path: pathlib.Path):

    # mesh axes
    # https://github.com/marcomusy/vedo/blob/56979bc2af270480fc8f2a6015cbf9773215ee39/vedo/addons.py#L1701
    undeformed_mesh_axes = dict(
        # general stuff
        titleFont='SmartCouric',
        labelFont="LogoType",
        axesLineWidth= 3,
        gridLineWidth= 2,
        xyPlaneColor='gray',
        xyGridColor='k1',
        # x axis details
        xtitle='x [m]', # latex-style syntax
        xrange=[0, 0.09],
        xValuesAndLabels=[(0,'0.00'),(0.01,'0.01'),(0.02,'0.02'),(0.03,'0.03'),
            (0.04,'0.04'), (0.05,'0.05'), (0.06,'0.06'), (0.07,'0.07'),
            (0.08,'0.08')],
        xTitlePosition=0.5,  # title fractional positions along axis
        xTitleJustify='top-center', # align title wrt to its axis
        xTitleSize=0.035,
        xTitleOffset=0.05,
        xLabelSize=0.025,
        # y axis details
        ytitle='y [m]',
        yrange=[0, 0.045],
        yValuesAndLabels=[(0,'0.00'),(0.01,'0.01'),(0.02,'0.02'),(0.03,'0.03'),
            (0.04,'0.04')],
        yTitlePosition=0.5,  # title fractional positions along axis
        yTitleJustify='top-center', # align title wrt to its axis
        yTitleSize=0.035,
        yTitleOffset=0.06,
        yLabelSize=0.025,    # size of the numeric labels along Y axis
        # background
        xyFrameColor='k1'
    )

    # undeformed mesh
    undeformed_mesh = FFD.input_mesh
    vedo_undeformed_mesh = undeformed_mesh.vedo_mesh
    vedo_undeformed_mesh.wireframe(True)
    vedo_undeformed_mesh.lineWidth(2)
    plot = vedo.show(vedo_undeformed_mesh, axes=undeformed_mesh_axes, 
        offscreen=True, size=(4800,3200))
    plot.screenshot(path)

# plot_undeformed_mesh

def plot_deformed_mesh(FFD: gus.FreeFormDeformation, path: pathlib.Path):

    # mesh axes
    # https://github.com/marcomusy/vedo/blob/56979bc2af270480fc8f2a6015cbf9773215ee39/vedo/addons.py#L1701
    deformed_mesh_axes = dict(
        # general stuff
        titleFont='SmartCouric',
        labelFont="LogoType",
        axesLineWidth= 3,
        gridLineWidth= 2,
        xyPlaneColor='gray',
        xyGridColor='k1',
        # x axis details
        xtitle='x [m]', # latex-style syntax
        xrange=[0, 0.09],
        xValuesAndLabels=[(0,'0.00'),(0.01,'0.01'),(0.02,'0.02'),(0.03,'0.03'),
            (0.04,'0.04'), (0.05,'0.05'), (0.06,'0.06'), (0.07,'0.07'),
            (0.08,'0.08')],
        xTitlePosition=0.5,  # title fractional positions along axis
        xTitleJustify='top-center', # align title wrt to its axis
        xTitleSize=0.035,
        xTitleOffset=0.05,
        xLabelSize=0.025,
        # y axis details
        ytitle='y [m]',
        yrange=[0, 0.05],
        yValuesAndLabels=[(0,'0.00'),(0.01,'0.01'),(0.02,'0.02'),(0.03,'0.03'),
            (0.04,'0.04')],
        yTitlePosition=0.5,  # title fractional positions along axis
        yTitleJustify='top-center', # align title wrt to its axis
        yTitleSize=0.035,
        yTitleOffset=0.06,
        yLabelSize=0.025,    # size of the numeric labels along Y axis
        # background
        xyFrameColor='k1'
    )

    # deformed mesh
    deformed_mesh = FFD.deformed_mesh
    vedo_deformed_mesh = deformed_mesh.vedo_mesh
    vedo_deformed_mesh.wireframe(True)
    vedo_deformed_mesh.lineWidth(2)
    plot = vedo.show(vedo_deformed_mesh, axes=deformed_mesh_axes, 
        offscreen=True, size=(4800,3200))
    plot.screenshot(path)

# plot_deformed_mesh

def plot_undeformed_spline(FFD: gus.FreeFormDeformation, path: pathlib.Path):

    undeformed_spline_axes = dict(
        # general stuff
        titleFont='SmartCouric',
        labelFont="LogoType",
        axesLineWidth= 4,
        gridLineWidth= 3,
        xyPlaneColor='gray',
        xyGridColor='k1',
        # x axis details
        xtitle='x [-]', # latex-style syntax
        xrange=[0, 1.05],
        xValuesAndLabels=[(0,'0.0'),(0.2,'0.2'),(0.4,'0.4'),(0.6,'0.6'),
            (0.8,'0.8'), (1.0,'1.0')],
        xTitlePosition=0.5,  # title fractional positions along axis
        xTitleJustify='top-center', # align title wrt to its axis
        xTitleSize=0.035,
        xTitleOffset=0.05,
        xLabelSize=0.025,
        # y axis details
        ytitle='y [-]',
        yrange=[0, 1.05],
        yValuesAndLabels=[(0,'0.0'),(0.2,'0.2'),(0.4,'0.4'),(0.6,'0.6'),
            (0.8,'0.8'), (1.0,'1.0')],
        yTitlePosition=0.5,  # title fractional positions along axis
        yTitleJustify='top-center', # align title wrt to its axis
        yTitleSize=0.035,
        yTitleOffset=0.12,
        yLabelSize=0.025,    # size of the numeric labels along Y axis
        # background
        xyFrameColor='k1'
    )

    vedo_undeformed_spline = FFD.undeformed_spline_.show(
        control_point_ids=False, offscreen=True
    )
    #vedo.show(vedo_undeformed_spline, axes=undeformed_spline_axes, 
    #   interactive=True).close()
    plot = vedo.show(vedo_undeformed_spline, axes=undeformed_spline_axes, 
        offscreen=True, size=(4800,3200))
    plot.screenshot(path)

# plot_undeformed_spline

def plot_deformed_spline(FFD: gus.FreeFormDeformation, path: pathlib.Path):

    deformed_spline_axes = dict(
        # general stuff
        titleFont='SmartCouric',
        labelFont="LogoType",
        axesLineWidth= 4,
        gridLineWidth= 3,
        xyPlaneColor='gray',
        xyGridColor='k1',
        # x axis details
        xtitle='x [-]', # latex-style syntax
        xrange=[0, 1.05],
        xValuesAndLabels=[(0,'0.0'),(0.2,'0.2'),(0.4,'0.4'),(0.6,'0.6'),
            (0.8,'0.8'), (1.0,'1.0')],
        xTitlePosition=0.5,  # title fractional positions along axis
        xTitleJustify='top-center', # align title wrt to its axis
        xTitleSize=0.035,
        xTitleOffset=0.05,
        xLabelSize=0.025,
        # y axis details
        ytitle='y [-]',
        yrange=[0, 1.25],
        yValuesAndLabels=[(0,'0.0'),(0.2,'0.2'),(0.4,'0.4'),(0.6,'0.6'),
            (0.8,'0.8'), (1.0,'1.0'), (1.2,'1.2')],
        yTitlePosition=0.5,  # title fractional positions along axis
        yTitleJustify='top-center', # align title wrt to its axis
        yTitleSize=0.035,
        yTitleOffset=0.12,
        yLabelSize=0.025,    # size of the numeric labels along Y axis
        # background
        xyFrameColor='k1'
    )

    vedo_deformed_spline = FFD.deformed_spline.show(
        control_point_ids=False, offscreen=True
    )
    #vedo.show(vedo_deformed_spline, axes=deformed_spline_axes, 
    #   interactive=True).close()
    plot = vedo.show(vedo_deformed_spline, axes=deformed_spline_axes, 
        offscreen=True, size=(4800,3200))
    plot.screenshot(path)
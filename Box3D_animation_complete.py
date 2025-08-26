
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.ioff()

# --- Figura y eje 3D ---
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# --- Utilidades de vista ---
def setaxis(x1, x2, y1, y2, z1, z2):
    ax.set_xlim3d(x1, x2)
    ax.set_ylim3d(y1, y2)
    ax.set_zlim3d(z1, z2)
    ax.view_init(elev=30, azim=40)

def set_equal_aspect():
    x_limits = np.array(ax.get_xlim3d())
    y_limits = np.array(ax.get_ylim3d())
    z_limits = np.array(ax.get_zlim3d())
    ranges = np.array([x_limits[1]-x_limits[0],
                       y_limits[1]-y_limits[0],
                       z_limits[1]-z_limits[0]])
    centers = np.array([x_limits.mean(), y_limits.mean(), z_limits.mean()])
    radius = 0.5 * ranges.max()
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

# --- Ejes fijos ---
def fix_system(axis_length=10, linewidth=2):
    ax.plot3D([0, axis_length], [0, 0], [0, 0], color='red', linewidth=linewidth)   # X
    ax.plot3D([0, 0], [0, axis_length], [0, 0], color='blue', linewidth=linewidth)  # Y
    ax.plot3D([0, 0], [0, 0], [0, axis_length], color='green', linewidth=linewidth) # Z

# --- Trig en grados ---
def sind(t): return np.sin(np.deg2rad(t))
def cosd(t): return np.cos(np.deg2rad(t))

# --- Matrices de rotación ---
def RotX(t):
    c, s = cosd(t), sind(t)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]])

def RotY(t):
    c, s = cosd(t), sind(t)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def RotZ(t):
    c, s = cosd(t), sind(t)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]])

# --- Dibujo ---
def drawVector(p_fin, p_init=(0,0,0), color='black', linewidth=1):
    x = [p_init[0], p_fin[0]]
    y = [p_init[1], p_fin[1]]
    z = [p_init[2], p_fin[2]]
    ax.plot3D(x, y, z, color=color, linewidth=linewidth)

def drawBox(pts8, color='black', linewidth=1.5):
    edges = [
        (0,1),(1,2),(2,3),(3,0), # base inferior
        (4,5),(5,6),(6,7),(7,4), # base superior
        (0,4),(1,5),(2,6),(3,7)  # verticales
    ]
    for i,j in edges:
        drawVector(pts8[j], pts8[i], color=color, linewidth=linewidth)

# --- Rotación de un conjunto de vértices ---
def apply_rotation(pts8, axis='z', angle=0):
    if axis == 'x':
        R = RotX(angle)
    elif axis == 'y':
        R = RotY(angle)
    else:
        R = RotZ(angle)
    return (R @ pts8.T).T

# --- Caja inicial ---
box_init = np.array([
    [0,0,0],
    [7,0,0],
    [7,0,3],
    [0,0,3],
    [0,2,0],
    [7,2,0],
    [7,2,3],
    [0,2,3]
], dtype=float)

def animate_box(box_current, axis='x', angle_to=40, angle_step=1, pause_s=0.02):
    angle = 0
    while angle <= angle_to:
        ax.cla()
        setaxis(-5,12,-5,12,-5,12)
        fix_system(axis_length=10, linewidth=2)

        box_rot = apply_rotation(box_current, axis=axis, angle=angle)
        drawBox(box_rot, color='purple', linewidth=2.0)


        plt.draw()
        plt.pause(pause_s)
        angle += angle_step

    # Al terminar, dejamos la caja en la orientación final
    return apply_rotation(box_current, axis=axis, angle=angle_to)

def run():
    box_after_x = animate_box(box_init, axis='x', angle_to=100)
    box_after_y = animate_box(box_after_x, axis='y', angle_to=40)
    box_after_z = animate_box(box_after_y, axis='z', angle_to=25)

    ax.cla()
    setaxis(-5,12,-5,12,-5,12)
    fix_system(axis_length=10, linewidth=2)
    drawBox(box_after_z, color='purple', linewidth=2.0)
    set_equal_aspect()

    plt.show()

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        sys.exit(0)

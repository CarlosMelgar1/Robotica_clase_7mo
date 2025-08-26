
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
    ax.plot3D([0, axis_length], [0, 0], [0, 0], color='red',   linewidth=linewidth)  # X
    ax.plot3D([0, 0], [0, axis_length], [0, 0], color='blue',  linewidth=linewidth)  # Y
    ax.plot3D([0, 0], [0, 0], [0, axis_length], color='green', linewidth=linewidth)  # Z

# --- Trig en grados ---
def sind(t): return np.sin(np.deg2rad(t))
def cosd(t): return np.cos(np.deg2rad(t))

# --- Matrices de rotación elementales ---
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

# --- Composición Euler (orden configurable) ---
def apply_rotation_euler(ax_deg, ay_deg, az_deg, order='xyz'):
    Rm = {'x': RotX, 'y': RotY, 'z': RotZ}
    angles = {'x': ax_deg, 'y': ay_deg, 'z': az_deg}
    R = np.eye(3)
    for axis in order:        # aplica en el orden izquierdo->derecho
        R = Rm[axis](angles[axis]) @ R
    return R

# --- Dibujo ---
def drawVector(p_fin, p_init=(0,0,0), color='black', linewidth=1):
    x = [p_init[0], p_fin[0]]
    y = [p_init[1], p_fin[1]]
    z = [p_init[2], p_fin[2]]
    ax.plot3D(x, y, z, color=color, linewidth=linewidth)

def drawBox(pts8, color, linewidth=2.0):
    edges = [
        (0,1),(1,2),(2,3),(3,0), # base inferior
        (4,5),(5,6),(6,7),(7,4), # base superior
        (0,4),(1,5),(2,6),(3,7)  # verticales
    ]
    for i,j in edges:
        drawVector(pts8[j], pts8[i], color=color, linewidth=linewidth)

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

# --- Animación simultánea ---
def animate_box_together(box_start,
                         target_ax=100, target_ay=40, target_az=25,
                         steps=180, pause_s=0.02, order='xyz'):

    for k in range(steps + 1):
        frac = k / steps  # 0 -> 1
        ax_deg = target_ax * frac
        ay_deg = target_ay * frac
        az_deg = target_az * frac

        R = apply_rotation_euler(ax_deg, ay_deg, az_deg, order=order)
        box_rot = (R @ box_start.T).T

        ax.cla()
        setaxis(-5,12,-5,12,-5,12)
        fix_system(axis_length=10, linewidth=2)
        drawBox(box_rot, color='black', linewidth=2.0)

        plt.draw()
        plt.pause(pause_s)

    return (R @ box_start.T).T

def run():
    # Rotación simultánea hacia los ángulos objetivo
    final_box = animate_box_together(
        box_init,
        target_ax=100, target_ay=40, target_az=25,   # ← tus ángulos destino
        steps=180,                                   # ← más pasos = animación más suave
        pause_s=0.02,                                # ← pausa entre frames
        order='xyz'                                  # ← orden de composición (importa)
    )

    # Mostrar posición final
    ax.cla()
    setaxis(-5,12,-5,12,-5,12)
    fix_system(axis_length=10, linewidth=2)
    drawBox(final_box, color='black', linewidth=2.0)
    # set_equal_aspect()
    plt.show()

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        sys.exit(0)

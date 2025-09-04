import matplotlib.pyplot as plt
import numpy as np

# --- Crear figura y eje 3D ---
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# --- Configuración de vista ---
def setaxis(x1, x2, y1, y2, z1, z2):
    ax.set_xlim3d(x1, x2)
    ax.set_ylim3d(y1, y2)
    ax.set_zlim3d(z1, z2)
    ax.view_init(elev=30, azim=40)

def set_equal_aspect():
    # Igualar escala en los tres ejes (cubo)
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

# --- Ejes de referencia ---
def fix_system(axis_length=10, linewidth=2):
    ax.plot3D([0, axis_length], [0, 0], [0, 0], color='red', linewidth=linewidth)   # X
    ax.plot3D([0, 0], [0, axis_length], [0, 0], color='blue', linewidth=linewidth)  # Y
    ax.plot3D([0, 0], [0, 0], [0, axis_length], color='green', linewidth=linewidth) # Z

# --- Seno y coseno en grados ---
def sind(t): return np.sin(np.deg2rad(t))
def cosd(t): return np.cos(np.deg2rad(t))

# --- Rotaciones 3x3 ---
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

# --- Matriz homogénea 4x4 (rotación + traslación) ---
def build_SE3(R=np.eye(3), t=(0,0,0)):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T

# --- Aplicar T a puntos ---
def apply_SE3(points, T):
    P = np.c_[points, np.ones(len(points))]   # paso a homogéneas
    P_t = (T @ P.T).T
    return P_t[:,:3]                         # regreso a 3D

# --- Funciones para dibujar ---
def drawVector(p_fin, p_init=(0,0,0), color='black', linewidth=1):
    x = [p_init[0], p_fin[0]]
    y = [p_init[1], p_fin[1]]
    z = [p_init[2], p_fin[2]]
    ax.plot3D(x, y, z, color=color, linewidth=linewidth)

def drawScatter(point, color='black', marker='o', s=20):
    ax.scatter(point[0], point[1], point[2], marker=marker, color=color, s=s)

def drawBox(pts8, color='black', show_points=True, linewidth=1.5):
    if show_points:
        for p in pts8:
            drawScatter(p, color=color, s=15)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i,j in edges:
        drawVector(pts8[j], pts8[i], color=color, linewidth=linewidth)

# --- Caja inicial ---
p1 = [0,0,0]
p2 = [7,0,0]
p3 = [7,0,3]
p4 = [0,0,3]
p5 = [0,2,0]
p6 = [7,2,0]
p7 = [7,2,3]
p8 = [0,2,3]
box_init = np.array([p1,p2,p3,p4,p5,p6,p7,p8], dtype=float)

# --- Composición de rotaciones ---
def compose_R(ax_deg=0, ay_deg=0, az_deg=0):
    return RotZ(az_deg) @ RotY(ay_deg) @ RotX(ax_deg)

# --- Animación de la caja ---
def animate_box(angle_to=25, angle_step=1, pause_s=0.02):
    angle = 0
    while angle <= angle_to:
        ax.cla()
        setaxis(-5,12,-5,12,-5,12)
        fix_system(axis_length=10, linewidth=2)

        R = compose_R(ay_deg=angle)   # rotación en Y
        t = (0,0,0)                   # traslación (aquí en cero)
        T = build_SE3(R, t)

        box_tf = apply_SE3(box_init, T)

        drawBox(box_tf, color='orange', show_points=False, linewidth=2.0)
        set_equal_aspect()
        plt.draw()
        plt.pause(pause_s)
        angle += angle_step

# --- Ejecutar ---
animate_box(angle_to=40, angle_step=1, pause_s=0.02)
plt.show()

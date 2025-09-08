import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

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

def fix_system(axis_length=8, linewidth=1.5):
    ax.plot3D([0, axis_length], [0, 0], [0, 0], color='red', linewidth=linewidth)
    ax.plot3D([0, 0], [0, axis_length], [0, 0], color='blue', linewidth=linewidth)
    ax.plot3D([0, 0], [0, 0], [0, axis_length], color='green', linewidth=linewidth)

def drawVector(p_fin, p_init=(0,0,0), color='black', linewidth=2.0):
    x = [p_init[0], p_fin[0]]
    y = [p_init[1], p_fin[1]]
    z = [p_init[2], p_fin[2]]
    ax.plot3D(x, y, z, color=color, linewidth=linewidth)

def drawMobileFrame(T, axis_scale=2.0, lw=2.0):
    R = T[:3, :3]
    o = T[:3, 3]
    ex = o + axis_scale * R[:, 0]
    ey = o + axis_scale * R[:, 1]
    ez = o + axis_scale * R[:, 2]
    drawVector(ex, o, color='red',  linewidth=lw)
    drawVector(ey, o, color='blue', linewidth=lw)
    drawVector(ez, o, color='green',linewidth=lw)

def sind(t): return np.sin(np.deg2rad(t))
def cosd(t): return np.cos(np.deg2rad(t))

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

def build_SE3(R=np.eye(3), t=(0,0,0)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = np.asarray(t).ravel()
    return T

def Tx(a):
    return build_SE3(np.eye(3), (a, 0, 0))

def Rz_SE3(theta):
    return build_SE3(RotZ(theta), (0,0,0))

def Ry_SE3(theta):
    return build_SE3(RotY(theta), (0,0,0))

def forward_frames(t1, t2, t3, l1, l2, l3, phi_y):
    T0  = np.eye(4)
    T01 = Rz_SE3(t1) @ Tx(l1)
    T02 = T01 @ (Rz_SE3(t2) @ Tx(l2))
    T03 = T02 @ (Rz_SE3(t3) @ Tx(l3))
    T_base = Ry_SE3(phi_y)
    G0 = T_base @ T0
    G1 = T_base @ T01
    G2 = T_base @ T02
    G3 = T_base @ T03
    return G0, G1, G2, G3

def draw_arm(frames, link_lw=4.0, frame_axis_scale=2.0, frame_lw=2.0):
    G0, G1, G2, G3 = frames
    o0, o1, o2, o3 = G0[:3,3], G1[:3,3], G2[:3,3], G3[:3,3]
    drawVector(o1, o0, color='black', linewidth=link_lw)
    drawVector(o2, o1, color='black', linewidth=link_lw)
    drawVector(o3, o2, color='black', linewidth=link_lw)
    drawMobileFrame(G0, axis_scale=frame_axis_scale, lw=frame_lw)
    drawMobileFrame(G1, axis_scale=frame_axis_scale, lw=frame_lw)
    drawMobileFrame(G2, axis_scale=frame_axis_scale, lw=frame_lw)
    drawMobileFrame(G3, axis_scale=frame_axis_scale, lw=frame_lw)

# longitudes de los eslabones
l1, l2, l3 = 7.5, 6.0, 4.5

# objetivos de rotación
t1_target, t2_target, t3_target = 30, 50, 25
phi_y_target = 40   # rotación global en Y

# velocidad
step_angle = 2.0
pause_s    = 0.02
world = 18

def redraw_scene(t1, t2, t3, phi_y):
    ax.cla()
    setaxis(-world, world, -world, world, -world, world)
    fix_system(axis_length=8, linewidth=1.5)
    frames = forward_frames(t1, t2, t3, l1, l2, l3, phi_y)
    draw_arm(frames, link_lw=4.0, frame_axis_scale=2.0, frame_lw=2.0)
    set_equal_aspect()
    plt.pause(pause_s)

def animate_rotations_only():
    t1 = t2 = t3 = 0.0
    phi_y = 0.0
    while t1 < t1_target:
        t1 = min(t1 + step_angle, t1_target)
        redraw_scene(t1, t2, t3, phi_y)
    while t2 < t2_target:
        t2 = min(t2 + step_angle, t2_target)
        redraw_scene(t1, t2, t3, phi_y)
    while t3 < t3_target:
        t3 = min(t3 + step_angle, t3_target)
        redraw_scene(t1, t2, t3, phi_y)
    while phi_y < phi_y_target:
        phi_y = min(phi_y + step_angle, phi_y_target)
        redraw_scene(t1, t2, t3, phi_y)
    plt.pause(0.5)

if __name__ == "__main__":
    setaxis(-world, world, -world, world, -world, world)
    fix_system(axis_length=8, linewidth=1.5)
    plt.draw()
    animate_rotations_only()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import time

# --- Configuración general ---
WORLD = 90.0         # Escena 3D
STEPS = 100
PAUSE = 0.02
OFFSET_Z = 1.0

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# --- Funciones auxiliares ---
def sind(t): return np.sin(np.deg2rad(t))
def cosd(t): return np.cos(np.deg2rad(t))

def A_DH(theta_deg, d, a, alpha_deg):
    ct, st = cosd(theta_deg), sind(theta_deg)
    ca, sa = cosd(alpha_deg), sind(alpha_deg)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,     sa,     ca,    d],
        [0,      0,      0,    1]
    ])

def drawVector(p_fin, p_init=(0,0,0), color='black', lw=2.0):
    ax.plot3D([p_init[0], p_fin[0]],
              [p_init[1], p_fin[1]],
              [p_init[2], p_fin[2]], color=color, linewidth=lw)

def drawMobileFrame(T, s=2.0):
    o = T[:3,3]
    R = T[:3,:3]
    drawVector(o + s*R[:,0], o, color='red')
    drawVector(o + s*R[:,1], o, color='blue')
    drawVector(o + s*R[:,2], o, color='green')

def set_scene():
    ax.cla()
    ax.set_xlim3d(-WORLD, WORLD)
    ax.set_ylim3d(-WORLD, WORLD)
    ax.set_zlim3d(0, WORLD)
    ax.view_init(elev=30, azim=45)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")


    # Ejes base
    ax.plot3D([0,15],[0,0],[0,0],'r')
    ax.plot3D([0,0],[0,15],[0,0],'b')
    ax.plot3D([0,0],[0,0],[15],'g')

# --- Cinemática directa SCARA 4DOF ---
def forward_kinematics_SCARA(theta1, theta2, d3, theta4, L1, L2):
    A1 = A_DH(theta1, 0, L1, 0)
    A2 = A_DH(theta2, 0, L2, 0)
    A3 = A_DH(0, -d3, 0, 0)     
    A4 = A_DH(theta4, 0, 0, 0)  

    T0 = np.eye(4)
    T1 = A1
    T2 = A1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    return [T0, T1, T2, T3, T4]

def draw_effector_cross(T, size=2.0):
    o = T[:3,3]
    R = T[:3,:3]
    p1x = o + R @ np.array([ size,  0, 0])
    p2x = o + R @ np.array([-size,  0, 0])
    p1y = o + R @ np.array([ 0,  size, 0])
    p2y = o + R @ np.array([ 0, -size, 0])
    drawVector(p1x, p2x, color='purple', lw=3)
    drawVector(p1y, p2y, color='purple', lw=3)

def draw_arm(frames):
    origins = [f[:3,3] for f in frames]
    drawVector(origins[1], origins[0], color='darkred', lw=6)
    drawVector(origins[2], origins[1], color='orange', lw=5)
    drawVector(origins[3], origins[2], color='gray', lw=4)
    drawVector(origins[4], origins[3], color='blue', lw=3)
    draw_effector_cross(frames[-1])
    for f in frames:
        drawMobileFrame(f)

def animate_to_target(t1_t, t2_t, d3_t, t4_t, L1, L2):
    t1_vals = np.linspace(0, t1_t, STEPS)
    t2_vals = np.linspace(0, t2_t, STEPS)
    d3_vals = np.linspace(0, d3_t, STEPS)
    t4_vals = np.linspace(0, t4_t, STEPS)

    for i in range(STEPS):
        set_scene()
        frames = forward_kinematics_SCARA(
            t1_vals[i], t2_vals[i], d3_vals[i], t4_vals[i],
            L1, L2
        )
        draw_arm(frames)
        plt.pause(PAUSE)

if __name__ == "__main__":

    L1 = 47.5
    L2 = 37.5

    t1_target = float(input("θ1 "))
    t2_target = float(input("θ2 "))
    d3_target = float(input("d3 "))
    t4_target = float(input("θ4 "))

    animate_to_target(t1_target, t2_target, d3_target, t4_target, L1, L2)
    plt.show()

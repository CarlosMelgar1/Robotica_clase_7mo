#   (Escribe 'q' para salir en cualquier entrada)

import math
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------ Utilidades ------------------

@dataclass
class IKResult:
    reachable: bool
    th1_deg: float | None
    th2_deg: float | None
    th3_deg: float | None
    message: str

def to_deg(a: float) -> float:
    d = math.degrees(a)
    while d <= -180: d += 360
    while d > 180: d -= 360
    return d

def linspace(a: float, b: float, n: int):
    if n <= 1: return [b]
    step = (b - a) / (n - 1)
    return [a + i*step for i in range(n)]

def leer_valor(msg: str):
    val = input(msg)
    if val.strip().lower() == 'q':
        return 'q'
    try:
        return float(val)
    except Exception:
        raise ValueError("Entrada no válida. Escribe un número o 'q' para salir.")

def leer_modo(msg: str, default="arriba"):
    v = input(msg).strip().lower()
    if v == 'q': return 'q'
    if v == '': return default
    if v in ('arriba','a','up') or 'arr' in v: return 'arriba'
    if v in ('abajo','b','down') or 'ab' in v: return 'abajo'
    return default

# ------------------ IK (RRR esférico) ------------------

def ik_rrr_spherical(L1: float, L2: float, x: float, y: float, z: float, elbow_mode: str) -> IKResult:
    """
    IK cerrada:
      th1 = atan2(y, x)
      En el plano (r,z) con r = sqrt(x^2+y^2):
        cos(th3) = (r^2 + z^2 - L1^2 - L2^2) / (2 L1 L2)
        th3 = atan2( ±sqrt(1-cos^2), cos )   (signo por codo arriba/abajo)
        th2 = atan2(z, r) - atan2(L2 sin(th3), L1 + L2 cos(th3))
    """
    if L1 <= 0 or L2 <= 0:
        return IKResult(False, None, None, None, "L1 y L2 deben ser positivos.")

    r = math.hypot(x, y)          # distancia en XY
    rho = math.hypot(r, z)        # distancia en el plano (r,z)

    # Alcanzabilidad del 2R
    if rho > L1 + L2 + 1e-9 or rho < abs(L1 - L2) - 1e-9:
        return IKResult(False, None, None, None, "Objetivo fuera del alcance del brazo.")

    th1 = math.atan2(y, x)

    # Ley de cosenos para el codo (th3)
    c3 = (r*r + z*z - L1*L1 - L2*L2) / (2.0 * L1 * L2)
    c3 = max(-1.0, min(1.0, c3))
    s3_abs = math.sqrt(max(0.0, 1.0 - c3*c3))
    s3 = +s3_abs if elbow_mode == 'arriba' else -s3_abs
    th3 = math.atan2(s3, c3)

    # Hombro (th2)
    k1 = L1 + L2 * c3
    k2 = L2 * s3
    th2 = math.atan2(z, r) - math.atan2(k2, k1)

    return IKResult(True, to_deg(th1), to_deg(th2), to_deg(th3), "OK")

# ------------------ FK ------------------

def fk_rrr_spherical(L1: float, L2: float, th1_deg: float, th2_deg: float, th3_deg: float):
    """
    FK:
      En el plano (r,z):
        r1 = L1 cos(th2), z1 = L1 sin(th2)
        r2 = r1 + L2 cos(th2+th3), z2 = z1 + L2 sin(th2+th3)
      Proyección a 3D con th1:
        x = r * cos(th1), y = r * sin(th1)
    """
    th1 = math.radians(th1_deg)
    th2 = math.radians(th2_deg)
    th3 = math.radians(th3_deg)

    r1 = L1 * math.cos(th2)
    z1 = L1 * math.sin(th2)
    r2 = r1 + L2 * math.cos(th2 + th3)
    z2 = z1 + L2 * math.sin(th2 + th3)

    x1 = r1 * math.cos(th1)
    y1 = r1 * math.sin(th1)
    x2 = r2 * math.cos(th1)
    y2 = r2 * math.sin(th1)

    base = (0.0, 0.0, 0.0)
    joint = (x1, y1, z1)
    tip   = (x2, y2, z2)
    return base, joint, tip

# ------------------ Animación ------------------

def animate_once_rrr(L1: float, L2: float, x_t: float, y_t: float, z_t: float,
                     elbow_mode: str = "arriba",
                     frames: int = 180, interval_ms: int = 20):
    """
    Trayectoria cartesiana lineal desde (L1+L2, 0, 0) hasta (x_t, y_t, z_t).
    Valida alcanzabilidad e IK en cada frame.
    """
    # Pose inicial: brazo extendido en +X, z=0
    x0, y0, z0 = (L1 + L2, 0.0, 0.0)

    xs = linspace(x0, x_t, frames)
    ys = linspace(y0, y_t, frames)
    zs = linspace(z0, z_t, frames)

    sols = []
    for xi, yi, zi in zip(xs, ys, zs):
        res = ik_rrr_spherical(L1, L2, xi, yi, zi, elbow_mode)
        if not res.reachable:
            raise RuntimeError(f"Trayectoria inalcanzable en ({xi:.3f},{yi:.3f},{zi:.3f}).")
        sols.append((res.th1_deg, res.th2_deg, res.th3_deg))

    # Inicialización con datos reales
    th1_0, th2_0, th3_0 = sols[0]
    base, joint, tip = fk_rrr_spherical(L1, L2, th1_0, th2_0, th3_0)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    link1, = ax.plot([base[0], joint[0]], [base[1], joint[1]], [base[2], joint[2]],
                     marker='o', linewidth=3)
    link2, = ax.plot([joint[0], tip[0]], [joint[1], tip[1]], [joint[2], tip[2]],
                     marker='o', linewidth=3)
    eff_scatter = ax.scatter([tip[0]], [tip[1]], [tip[2]], s=60, c='r')  # efector
    target_scatter = ax.scatter([x_t], [y_t], [z_t], marker='x', s=80)

    # Límites de la escena
    reach = L1 + L2
    m = max(reach, abs(x_t), abs(y_t), abs(z_t)) + 5.0
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"RRR esférico (parado) - 3D - Codo {elbow_mode}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.grid(True)
    ax.view_init(elev=25, azim=-45)

    def update(frame):
        th1, th2, th3 = sols[frame]
        base, joint, tip = fk_rrr_spherical(L1, L2, th1, th2, th3)
        link1.set_data_3d([base[0], joint[0]], [base[1], joint[1]], [base[2], joint[2]])
        link2.set_data_3d([joint[0], tip[0]], [joint[1], tip[1]], [joint[2], tip[2]])
        eff_scatter._offsets3d = ([tip[0]], [tip[1]], [tip[2]])
        return link1, link2, eff_scatter, target_scatter

    anim = FuncAnimation(fig, update, frames=len(sols), interval=interval_ms,
                         blit=False, repeat=False)
    fig._anim = anim   # mantener referencia
    plt.show()

# ------------------ Loop interactivo ------------------

def main():
    print("=== RRR esférico (parado) - Animación 3D ===")
    print("Escribe 'q' en cualquier entrada para salir.\n")
    while True:
        try:
            modo = leer_modo("Modo (arriba/abajo) [arriba]: ", default="arriba")
            if modo == 'q': break

            L1 = leer_valor("Ingresa L1: ")
            if L1 == 'q': break
            L2 = leer_valor("Ingresa L2: ")
            if L2 == 'q': break
            x  = leer_valor("Ingresa X del objetivo: ")
            if x == 'q': break
            y  = leer_valor("Ingresa Y del objetivo: ")
            if y == 'q': break
            z  = leer_valor("Ingresa Z del objetivo: ")
            if z == 'q': break

            L1, L2, x, y, z = float(L1), float(L2), float(x), float(y), float(z)

            try:
                animate_once_rrr(L1, L2, x, y, z, elbow_mode=modo, frames=180, interval_ms=20)
            except Exception as e:
                sys.stderr.write(f"[ERROR] {e}\n\n")  # no detiene el loop

        except ValueError as ve:
            sys.stderr.write(f"[ERROR] {ve}\n\n")
            continue
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break

if __name__ == "__main__":
    main()

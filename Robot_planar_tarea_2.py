#   (Escribe 'q' para salir en cualquier entrada)

import math
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------ IK (plano YZ) ------------------

@dataclass
class IKResult:
    reachable: bool
    theta1_deg: float | None
    theta2_deg: float | None
    message: str

def _to_deg(a: float) -> float:
    d = math.degrees(a)
    while d <= -180: d += 360
    while d > 180: d -= 360
    return d

def _ik_2r_yz(L1: float, L2: float, y: float, z: float, elbow: str) -> IKResult:
    """
    Inversa para 2R planar en el plano YZ (X=0).
    elbow: 'arriba' => sin(theta2) positivo, 'abajo' => sin(theta2) negativo.
    """
    if L1 <= 0 or L2 <= 0:
        return IKResult(False, None, None, "L1 y L2 deben ser positivos.")

    r2 = y*y + z*z
    r = math.sqrt(r2)
    tol = 1e-9
    if r > L1 + L2 + tol or r < abs(L1 - L2) - tol:
        return IKResult(False, None, None, "Objetivo fuera del alcance.")

    c2 = (r2 - L1*L1 - L2*L2) / (2.0 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    s2_abs = math.sqrt(max(0.0, 1.0 - c2*c2))
    s2 = +s2_abs if elbow.startswith("arr") else -s2_abs
    theta2 = math.atan2(s2, c2)

    k1 = L1 + L2 * c2
    k2 = L2 * s2
    # En YZ, el "atan2" de orientación usa (z, y)
    theta1 = math.atan2(z, y) - math.atan2(k2, k1)

    return IKResult(True, _to_deg(theta1), _to_deg(theta2), "OK")

# ------------------ FK (plano YZ) ------------------

def fk_2r_yz(L1: float, L2: float, th1_deg: float, th2_deg: float):
    """ Cinemática directa en plano YZ (X=0). """
    th1 = math.radians(th1_deg)
    th2 = math.radians(th2_deg)
    x0, y0, z0 = 0.0, 0.0, 0.0
    y1 = L1 * math.cos(th1)
    z1 = L1 * math.sin(th1)
    x1 = 0.0
    y2 = y1 + L2 * math.cos(th1 + th2)
    z2 = z1 + L2 * math.sin(th1 + th2)
    x2 = 0.0
    return (x0, y0, z0), (x1, y1, z1), (x2, y2, z2)

# ------------------ Utilidades ------------------

def linspace(a: float, b: float, n: int):
    if n <= 1:
        return [b]
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
    """
    Acepta: 'arriba', 'abajo', 'a', 'b', 'up', 'down'. Default 'arriba'.
    Si entra basura (como rutas/paths), usa default silenciosamente.
    """
    val = input(msg)
    if val.strip().lower() == 'q':
        return 'q'
    v = val.strip().lower()
    if v == '':
        return default
    if v in ('arriba', 'a', 'up') or 'arr' in v:
        return 'arriba'
    if v in ('abajo', 'b', 'down') or 'ab' in v:
        return 'abajo'
    # No molestar con mensajes: usa default
    return default

# ------------------ Animación (robusta) ------------------

def animate_once_yz(L1: float, L2: float, y_target: float, z_target: float,
                    elbow_mode: str = "arriba",
                    frames: int = 150, interval_ms: int = 20):
    """
    Anima desde brazo extendido sobre +Y hasta (y_target, z_target) en YZ.
    """
    if L1 <= 0 or L2 <= 0:
        raise ValueError("L1 y L2 deben ser positivos.")

    # Pose inicial: efector en +Y al alcance máximo
    y_start, z_start = (L1 + L2, 0.0)

    ys = linspace(y_start, y_target, frames)
    zs = linspace(z_start, z_target, frames)

    # Precalcular IK y validar
    thetas = []
    for yi, zi in zip(ys, zs):
        res = _ik_2r_yz(L1, L2, yi, zi, elbow=elbow_mode)
        if not res.reachable:
            raise RuntimeError(f"Trayectoria inalcanzable en ({yi:.3f}, {zi:.3f}).")
        thetas.append((res.theta1_deg, res.theta2_deg))

    # Inicialización con datos reales para evitar problemas de render
    th1_0, th2_0 = thetas[0]
    (x0,y0,z0), (x1,y1,z1), (x2,y2,z2) = fk_2r_yz(L1, L2, th1_0, th2_0)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    link1, = ax.plot([x0, x1], [y0, y1], [z0, z1], marker='o', linewidth=3)
    link2, = ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', linewidth=3)
    eff_scatter = ax.scatter([x2], [y2], [z2], s=60, c='r')  # punto rojo actual
    target_scatter = ax.scatter([0.0], [y_target], [z_target], marker='x', s=80)

    # Límites y vista — X estrecho (plano YZ)
    reach = L1 + L2
    m = max(reach, abs(y_target), abs(z_target)) + 5.0
    ax.set_xlim(-m*0.15, m*0.15)   # X casi fijo
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    ax.set_box_aspect((0.15, 1, 1))
    ax.set_title(f"2R Planar (PARADO en YZ) - Codo {elbow_mode} - 3D")
    ax.set_xlabel("X (≈0)"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.grid(True)
    ax.view_init(elev=25, azim=-35)

    def update(frame):
        th1, th2 = thetas[frame]
        (x0,y0,z0), (x1,y1,z1), (x2,y2,z2) = fk_2r_yz(L1, L2, th1, th2)
        link1.set_data_3d([x0, x1], [y0, y1], [z0, z1])
        link2.set_data_3d([x1, x2], [y1, y2], [z1, z2])
        eff_scatter._offsets3d = ([x2], [y2], [z2])
        return link1, link2, eff_scatter, target_scatter

    anim = FuncAnimation(fig, update, frames=len(thetas),
                         interval=interval_ms, blit=False, repeat=False)
    fig._anim = anim  # Mantener referencia
    plt.show()

# ------------------ Loop interactivo ------------------

def main():
    print("=== Animación 2R Planar PARADO (YZ) - Codo ARRIBA/ABAJO ===")
    print("Escribe 'q' en cualquier entrada para salir.\n")
    while True:
        try:
            modo = leer_modo("Modo (arriba/abajo) [arriba]: ", default="arriba")
            if modo == 'q':
                break

            L1 = leer_valor("Ingresa L1 (longitud del primer eslabón): ")
            if L1 == 'q': break
            L2 = leer_valor("Ingresa L2 (longitud del segundo eslabón): ")
            if L2 == 'q': break
            y  = leer_valor("Ingresa Y del objetivo: ")
            if y == 'q': break
            z  = leer_valor("Ingresa Z del objetivo: ")
            if z == 'q': break

            L1, L2, y, z = float(L1), float(L2), float(y), float(z)

            try:
                animate_once_yz(L1, L2, y, z, elbow_mode=modo, frames=150, interval_ms=20)
            except Exception as e:
                # No detener el programa: solo mostrar error y continuar el loop
                sys.stderr.write(f"[ERROR] {e}\n\n")

        except ValueError as ve:
            sys.stderr.write(f"[ERROR] {ve}\n\n")
            continue
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break

if __name__ == "__main__":
    main()

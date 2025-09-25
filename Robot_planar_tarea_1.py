#   (Escribe 'q' para salir en cualquier entrada)

import math
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (habilita proyección 3D)

# ------------------ IK ------------------

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

def _ik_2r(L1: float, L2: float, x: float, y: float, elbow: str) -> IKResult:
    """
    Inversa 2R planar.
    elbow: "arriba" => seno(theta2) positivo
           "abajo"  => seno(theta2) negativo
    """
    r2 = x*x + y*y
    r = math.sqrt(r2)
    tol = 1e-9

    if L1 <= 0 or L2 <= 0:
        return IKResult(False, None, None, "L1 y L2 deben ser positivos.")
    if r > L1 + L2 + tol or r < abs(L1 - L2) - tol:
        return IKResult(False, None, None, "Objetivo fuera del alcance geométrico del robot.")

    # Ley de cosenos
    c2 = (r2 - L1*L1 - L2*L2) / (2.0 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    s2_abs = math.sqrt(max(0.0, 1.0 - c2*c2))

    if elbow.lower().startswith("arr"):   # codo ARRIBA
        s2 = +s2_abs
    else:                                 # codo ABAJO
        s2 = -s2_abs

    theta2 = math.atan2(s2, c2)

    k1 = L1 + L2 * c2
    k2 = L2 * s2
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)

    return IKResult(True, _to_deg(theta1), _to_deg(theta2), "OK")

def ik_2r_elbow_up(L1, L2, x, y):   # helper explícito
    return _ik_2r(L1, L2, x, y, elbow="arriba")

def ik_2r_elbow_down(L1, L2, x, y): # helper explícito
    return _ik_2r(L1, L2, x, y, elbow="abajo")

# ------------------ FK y utilidades ------------------

def fk_2r(L1: float, L2: float, th1_deg: float, th2_deg: float):
    th1 = math.radians(th1_deg)
    th2 = math.radians(th2_deg)
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1 = L1 * math.cos(th1)
    y1 = L1 * math.sin(th1)
    z1 = 0.0
    x2 = x1 + L2 * math.cos(th1 + th2)
    y2 = y1 + L2 * math.sin(th1 + th2)
    z2 = 0.0
    return (x0, y0, z0), (x1, y1, z1), (x2, y2, z2)

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
    val = input(msg).strip().lower()
    if val == 'q':
        return 'q'
    if val == '':
        return default
    if val.startswith('arr'):
        return 'arriba'
    if val.startswith('ab'):
        return 'abajo'
    print("Modo no reconocido. Usando 'arriba'.")
    return 'arriba'

# ------------------ Animación (versión robusta) ------------------

def animate_once(L1: float, L2: float, x_target: float, y_target: float,
                 elbow_mode: str = "arriba",
                 frames: int = 150, interval_ms: int = 20):
    """
    Anima desde brazo extendido en +X hasta (x_target, y_target) con la solución indicada.
    elbow_mode: 'arriba' o 'abajo'
    En 3D (z=0). Si algún punto de la trayectoria es inalcanzable, lanza RuntimeError.
    """
    if L1 <= 0 or L2 <= 0:
        raise ValueError("L1 y L2 deben ser positivos.")

    # Pose inicial: efector en +X al alcance máximo
    x_start, y_start = (L1 + L2, 0.0)

    # Trayectoria cartesiana recta
    xs = linspace(x_start, x_target, frames)
    ys = linspace(y_start, y_target, frames)

    # Precalcular IK y validar alcanzabilidad
    thetas = []
    for xi, yi in zip(xs, ys):
        res = _ik_2r(L1, L2, xi, yi, elbow=elbow_mode)
        if not res.reachable:
            raise RuntimeError(
                f"Trayectoria inalcanzable: el punto ({xi:.3f}, {yi:.3f}) no es alcanzable."
            )
        thetas.append((res.theta1_deg, res.theta2_deg))

    # Posición inicial para inicializar líneas (evita que no se dibujen)
    th1_0, th2_0 = thetas[0]
    (x0,y0,z0), (x1,y1,z1), (x2,y2,z2) = fk_2r(L1, L2, th1_0, th2_0)

    # --- Figura 3D ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Elementos gráficos (inicializados con datos REALES)
    link1, = ax.plot([x0, x1], [y0, y1], [z0, z1], marker='o', linewidth=3)
    link2, = ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', linewidth=3)
    eff_scatter = ax.scatter([x2], [y2], [z2], s=60, c='r')  # punto rojo actual
    target_scatter = ax.scatter([x_target], [y_target], [0.0], marker='x', s=80)

    # Límites y vista
    reach = L1 + L2
    m = max(reach, abs(x_target), abs(y_target)) + 5.0
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_zlim(-m*0.2, m*0.2)
    ax.set_box_aspect((1, 1, 0.2))
    ax.set_title(f"2R Planar (acostado) - Codo {elbow_mode} - 3D")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (0)")
    ax.grid(True)
    ax.view_init(elev=25, azim=-55)

    def update(frame):
        th1_deg, th2_deg = thetas[frame]
        (x0,y0,z0), (x1,y1,z1), (x2,y2,z2) = fk_2r(L1, L2, th1_deg, th2_deg)

        link1.set_data_3d([x0, x1], [y0, y1], [z0, z1])
        link2.set_data_3d([x1, x2], [y1, y2], [z1, z2])
        eff_scatter._offsets3d = ([x2], [y2], [z2])  # punto rojo

        return link1, link2, eff_scatter, target_scatter

    # Mantener referencia viva para evitar garbage collection
    anim = FuncAnimation(fig, update, frames=len(thetas),
                         interval=interval_ms, blit=False, repeat=False)
    fig._anim = anim

    plt.show()

# ------------------ Loop interactivo ------------------

def main():
    print("=== Animación 2R Planar (acostado) - Codo ARRIBA/ABAJO - 3D ===")
    print("Escribe 'q' en cualquier entrada para salir.\n")

    while True:
        try:
            modo = leer_modo("Modo (arriba/abajo) [arriba]: ", default="arriba")
            if modo == 'q': break

            L1 = leer_valor("Ingresa L1 (longitud del primer eslabón): ")
            if L1 == 'q': break
            L2 = leer_valor("Ingresa L2 (longitud del segundo eslabón): ")
            if L2 == 'q': break
            x  = leer_valor("Ingresa x del objetivo: ")
            if x == 'q': break
            y  = leer_valor("Ingresa y del objetivo: ")
            if y == 'q': break

            L1 = float(L1); L2 = float(L2); x = float(x); y = float(y)

            try:
                animate_once(L1, L2, x, y, elbow_mode=modo, frames=150, interval_ms=20)
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

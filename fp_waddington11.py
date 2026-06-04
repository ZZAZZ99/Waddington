#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolución numérica de la ecuación de Fokker--Planck para el modelo de Hill
simétrico del paisaje de Waddington.

Este script está pensado para la Tarea 2 del trabajo:
- define el campo determinista F=(F1,F2);
- construye una malla uniforme en el dominio [xmin,xmax]^2;
- evoluciona la ecuación de Fokker--Planck mediante diferencias finitas;
- impone condiciones de contorno de Dirichlet;
- normaliza la densidad de probabilidad;
- calcula el potencial efectivo U=-log(P+delta);
- guarda figuras y diagnósticos.

Versión con malla N=150 y salida isométrica del potencial. No modifica el archivo .tex.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

try:
    from scipy.optimize import root
except ImportError:  # permite usar el script aunque scipy no esté disponible
    root = None


@dataclass
class ModelParams:
    """Parámetros del modelo de Hill simétrico."""
    a: float = 1.48
    b: float = 1.0
    k: float = 1.0
    S: float = 0.5
    n: int = 4
    D: float = 0.05


@dataclass
class GridParams:
    """Parámetros de discretización espacial."""
    xmin: float = -0.25
    xmax: float = 3.0
    N: int = 150  # número de intervalos; la malla tiene N+1 puntos por eje


@dataclass
class SolverParams:
    """Parámetros del integrador temporal."""
    safety: float = 0.20
    max_steps: int = 30000
    check_every: int = 100
    tol: float = 1.0e-7
    sigma: float = 0.16
    init_mode: str = "central"  # "central", "stable_fixed_points" o "symmetric_mixture"
    clip_negative: bool = True


def hill_field(
    X1: np.ndarray,
    X2: np.ndarray,
    params: ModelParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Campo determinista F=(F1,F2) del modelo de Hill simétrico.

    F1 = a*x1^n/(S^n+x1^n) + b*S^n/(S^n+x2^n) - k*x1
    F2 = a*x2^n/(S^n+x2^n) + b*S^n/(S^n+x1^n) - k*x2
    """
    Sn = params.S ** params.n
    X1n = X1 ** params.n
    X2n = X2 ** params.n

    F1 = params.a * X1n / (Sn + X1n) + params.b * Sn / (Sn + X2n) - params.k * X1
    F2 = params.a * X2n / (Sn + X2n) + params.b * Sn / (Sn + X1n) - params.k * X2
    return F1, F2


def make_grid(grid: GridParams) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Construye una malla uniforme.

    Convención: N es el número de intervalos, por lo que hay N+1 puntos.
    h=(xmax-xmin)/N.
    """
    x = np.linspace(grid.xmin, grid.xmax, grid.N + 1)
    h = (grid.xmax - grid.xmin) / grid.N
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    return x, h, X1, X2


def normalize_probability(P: np.ndarray, h: float) -> np.ndarray:
    """Normaliza P para que sum(P)*h^2=1."""
    mass = float(np.sum(P) * h * h)
    if not np.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("La masa de probabilidad no es positiva; el esquema se ha vuelto inestable.")
    return P / mass


def apply_dirichlet_boundary(P: np.ndarray) -> None:
    """Impone P=0 en la frontera del dominio."""
    P[0, :] = 0.0
    P[-1, :] = 0.0
    P[:, 0] = 0.0
    P[:, -1] = 0.0


def gaussian_2d(X1: np.ndarray, X2: np.ndarray, mu: Tuple[float, float], sigma: float) -> np.ndarray:
    """Gaussiana bidimensional no normalizada."""
    return np.exp(-((X1 - mu[0]) ** 2 + (X2 - mu[1]) ** 2) / (2.0 * sigma ** 2))


def numerical_jacobian(x: np.ndarray, params: ModelParams, eps: float = 1.0e-6) -> np.ndarray:
    """Jacobiano numérico del campo de Hill en un punto."""
    x = np.asarray(x, dtype=float)

    def f(z: np.ndarray) -> np.ndarray:
        F1, F2 = hill_field(np.array(z[0]), np.array(z[1]), params)
        return np.array([float(F1), float(F2)])

    J = np.zeros((2, 2), dtype=float)
    for j in range(2):
        dz = np.zeros(2)
        dz[j] = eps
        J[:, j] = (f(x + dz) - f(x - dz)) / (2.0 * eps)
    return J


def classify_fixed_point(point: np.ndarray, params: ModelParams) -> str:
    """Clasifica localmente un punto fijo por los autovalores del jacobiano."""
    eigvals = np.linalg.eigvals(numerical_jacobian(point, params))
    real_parts = np.real(eigvals)
    if np.all(real_parts < 0.0):
        return "stable"
    if np.any(real_parts > 0.0) and np.any(real_parts < 0.0):
        return "saddle"
    if np.all(real_parts > 0.0):
        return "unstable"
    return "nonhyperbolic"


def find_fixed_points(params: ModelParams, grid: GridParams) -> List[Dict[str, object]]:
    """
    Busca puntos fijos con scipy.root desde varias semillas.

    La función se usa para diagnóstico y para condiciones iniciales si
    init_mode='stable_fixed_points'. Si scipy no está disponible, devuelve [].
    """
    if root is None:
        return []

    def fun(z: np.ndarray) -> np.ndarray:
        F1, F2 = hill_field(np.array(z[0]), np.array(z[1]), params)
        return np.array([float(F1), float(F2)])

    guesses = [
        (0.0, params.a + params.b),
        (params.a + params.b, 0.0),
        (params.a, params.a),
        (0.0, 0.0),
        (0.2, 2.5),
        (2.5, 0.2),
        (1.0, 1.0),
        (2.0, 2.0),
        (0.5, 0.5),
        (0.5, 1.5),
        (1.5, 0.5),
    ]

    points: List[np.ndarray] = []
    for guess in guesses:
        sol = root(fun, np.array(guess, dtype=float), method="hybr")
        if not sol.success:
            continue
        p = sol.x
        if not (grid.xmin <= p[0] <= grid.xmax and grid.xmin <= p[1] <= grid.xmax):
            continue
        if np.linalg.norm(fun(p)) > 1.0e-7:
            continue
        if all(np.linalg.norm(p - q) > 1.0e-4 for q in points):
            points.append(p)

    out = []
    for p in sorted(points, key=lambda z: (z[0], z[1])):
        out.append(
            {
                "x1": float(p[0]),
                "x2": float(p[1]),
                "type": classify_fixed_point(p, params),
            }
        )
    return out


def initial_probability(
    X1: np.ndarray,
    X2: np.ndarray,
    h: float,
    params: ModelParams,
    grid: GridParams,
    solver: SolverParams,
) -> np.ndarray:
    """
    Construye la condición inicial.

    central:
        gaussiana centrada cerca de la rama simétrica x1=x2.
    stable_fixed_points:
        suma de gaussianas en puntos fijos estables calculados numéricamente.
    symmetric_mixture:
        mezcla simétrica orientativa: centro + dos estados diferenciados laterales.
    """
    if solver.init_mode == "central":
        center = (max(grid.xmin, min(params.a, grid.xmax)), max(grid.xmin, min(params.a, grid.xmax)))
        P = gaussian_2d(X1, X2, center, solver.sigma)

    elif solver.init_mode == "stable_fixed_points":
        fixed_points = find_fixed_points(params, grid)
        stable = [p for p in fixed_points if p["type"] == "stable"]
        if not stable:
            raise RuntimeError("No se encontraron puntos fijos estables para inicializar P.")
        P = np.zeros_like(X1)
        for p in stable:
            P += gaussian_2d(X1, X2, (float(p["x1"]), float(p["x2"])), solver.sigma)

    elif solver.init_mode == "symmetric_mixture":
        # Mezcla no finalista: útil para pruebas rápidas y para poblar regiones laterales.
        central = (max(grid.xmin, min(params.a, grid.xmax)), max(grid.xmin, min(params.a, grid.xmax)))
        lateral_high = max(grid.xmin, min(params.a + params.b, grid.xmax))
        low = max(grid.xmin, min(0.05, grid.xmax))
        P = (
            gaussian_2d(X1, X2, central, solver.sigma)
            + 0.35 * gaussian_2d(X1, X2, (low, lateral_high), solver.sigma)
            + 0.35 * gaussian_2d(X1, X2, (lateral_high, low), solver.sigma)
        )

    else:
        raise ValueError(f"init_mode no reconocido: {solver.init_mode}")

    apply_dirichlet_boundary(P)
    if solver.clip_negative:
        P = np.maximum(P, 0.0)
    return normalize_probability(P, h)


def choose_time_step(F1: np.ndarray, F2: np.ndarray, h: float, params: ModelParams, solver: SolverParams) -> Dict[str, float]:
    """Calcula un paso temporal explícito con una cota tipo CFL."""
    max_speed = float(np.max(np.abs(F1) + np.abs(F2)))
    diffusion_bound = h * h / (4.0 * params.D)
    advection_bound = np.inf if max_speed == 0.0 else h / max_speed
    dt = solver.safety * min(diffusion_bound, advection_bound)
    return {
        "dt": float(dt),
        "max_speed": max_speed,
        "diffusion_bound": float(diffusion_bound),
        "advection_bound": float(advection_bound),
        "safety": float(solver.safety),
    }


def fp_rhs_centered(P: np.ndarray, F1: np.ndarray, F2: np.ndarray, params: ModelParams, h: float) -> np.ndarray:
    """
    Lado derecho de Fokker--Planck en forma conservativa:

        dP/dt = -d(F1 P)/dx1 - d(F2 P)/dx2 + D Laplaciano(P).

    Se usan diferencias centradas para los términos espaciales interiores.
    """
    G1 = F1 * P
    G2 = F2 * P

    rhs = np.zeros_like(P)

    rhs[1:-1, 1:-1] = (
        - (G1[2:, 1:-1] - G1[:-2, 1:-1]) / (2.0 * h)
        - (G2[1:-1, 2:] - G2[1:-1, :-2]) / (2.0 * h)
        + params.D
        * (
            (P[2:, 1:-1] - 2.0 * P[1:-1, 1:-1] + P[:-2, 1:-1]) / (h * h)
            + (P[1:-1, 2:] - 2.0 * P[1:-1, 1:-1] + P[1:-1, :-2]) / (h * h)
        )
    )

    return rhs


def evolve_fokker_planck(
    params: ModelParams,
    grid: GridParams,
    solver: SolverParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Evoluciona la ecuación hasta alcanzar el criterio de convergencia o max_steps."""
    x, h, X1, X2 = make_grid(grid)
    F1, F2 = hill_field(X1, X2, params)
    dt_info = choose_time_step(F1, F2, h, params, solver)

    P = initial_probability(X1, X2, h, params, grid, solver)
    diagnostics: Dict[str, object] = {
        "model": asdict(params),
        "grid": asdict(grid),
        "solver": asdict(solver),
        "time_step": dt_info,
        "fixed_points": find_fixed_points(params, grid),
        "history": [],
    }

    converged = False
    last_delta = np.inf
    min_before_clip = 0.0

    for step in range(1, solver.max_steps + 1):
        P_old = P
        P_new = P_old + dt_info["dt"] * fp_rhs_centered(P_old, F1, F2, params, h)
        apply_dirichlet_boundary(P_new)

        min_before_clip = float(np.min(P_new))
        if solver.clip_negative:
            # No debe ocultar inestabilidades fuertes: el mínimo se guarda en diagnósticos.
            P_new = np.maximum(P_new, 0.0)

        P_new = normalize_probability(P_new, h)
        P = P_new

        if step % solver.check_every == 0 or step == 1:
            last_delta = float(np.max(np.abs(P - P_old)))
            mass = float(np.sum(P) * h * h)
            argmax = np.unravel_index(np.argmax(P), P.shape)
            diagnostics["history"].append(
                {
                    "step": step,
                    "delta_max": last_delta,
                    "mass": mass,
                    "min_P": float(np.min(P)),
                    "min_before_clip": min_before_clip,
                    "max_P": float(np.max(P)),
                    "argmax_x1": float(X1[argmax]),
                    "argmax_x2": float(X2[argmax]),
                }
            )
            if last_delta < solver.tol:
                converged = True
                break

    diagnostics["final"] = {
        "converged": converged,
        "steps": step,
        "delta_max": last_delta,
        "mass": float(np.sum(P) * h * h),
        "min_P": float(np.min(P)),
        "min_before_clip": min_before_clip,
        "max_P": float(np.max(P)),
    }

    argmax = np.unravel_index(np.argmax(P), P.shape)
    diagnostics["final"]["argmax_x1"] = float(X1[argmax])
    diagnostics["final"]["argmax_x2"] = float(X2[argmax])

    return P, X1, X2, diagnostics


def effective_potential(P: np.ndarray, relative_delta: float = 1.0e-12) -> np.ndarray:
    """Calcula U=-log(P+delta) y desplaza el mínimo a cero para graficar."""
    delta = relative_delta * float(np.max(P))
    U = -np.log(P + delta)
    U = U - np.min(U)
    return U


def a_tag(a: float) -> str:
    """Etiqueta de a para nombres de archivo: 1.48 -> a148, 0.10 -> a010."""
    return f"a{int(round(100 * a)):03d}"


def plot_density(
    Z: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    title: str,
    cbar_label: str,
    output_path: Path,
) -> None:
    """Guarda un mapa 2D de una magnitud definida en la malla."""
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    image = ax.pcolormesh(X1, X2, Z, shading="auto")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)




def plot_potential_isometric(
    U: np.ndarray,
    P: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    title: str,
    output_path: Path,
    p_floor_rel: float = 1.0e-2,
    relevant_quantile: float = 0.99,
    elev: float = 25.0,
    azim: float = -60.0,
    z_depth: float = 1.25,
    visual_xlim: Tuple[float, float] = (0.0, 3.0),
    visual_ylim: Tuple[float, float] = (0.0, 3.0),
    show_colorbar: bool = True,
    show_title: bool = False,
) -> None:
    """
    Guarda una vista isométrica del paisaje efectivo U(x1,x2).

    Idea clave:
    la frontera de Dirichlet hace que P≈0 en los bordes, por lo que
    U=-ln(P+δ) crece muchísimo allí y aplana la figura si se usa todo el
    rango de U. Para evitarlo, el recorte visual no se calcula con todos
    los puntos de la malla, sino solo con la región probabilísticamente
    relevante, definida por P >= p_floor_rel * max(P) dentro de la ventana
    representada.

    Con ello se conserva la estructura de los mínimos internos del paisaje
    (central y laterales) sin dejar que la frontera domine la escala.
    La transformación visual sigue siendo monótona y lineal:

        Z = -z_depth * (1 - min(U, U_vis_max)/U_vis_max),

    donde U_vis_max se obtiene a partir de los puntos relevantes.
    """
    if not (0.0 < p_floor_rel < 1.0):
        raise ValueError("p_floor_rel debe pertenecer al intervalo (0,1).")
    if not (0.0 < relevant_quantile <= 1.0):
        raise ValueError("relevant_quantile debe pertenecer al intervalo (0,1].")
    if z_depth <= 0.0:
        raise ValueError("z_depth debe ser positivo.")

    x1_axis = X1[:, 0]
    x2_axis = X2[0, :]
    i_sel = np.where((x1_axis >= visual_xlim[0]) & (x1_axis <= visual_xlim[1]))[0]
    j_sel = np.where((x2_axis >= visual_ylim[0]) & (x2_axis <= visual_ylim[1]))[0]
    if len(i_sel) == 0 or len(j_sel) == 0:
        raise RuntimeError("La ventana visual no contiene puntos de la malla.")

    X1v = X1[np.ix_(i_sel, j_sel)]
    X2v = X2[np.ix_(i_sel, j_sel)]
    Uv = U[np.ix_(i_sel, j_sel)]
    Pv = P[np.ix_(i_sel, j_sel)]

    pmax = float(np.nanmax(Pv))
    if not np.isfinite(pmax) or pmax <= 0.0:
        raise RuntimeError("No se puede construir la figura: max(P) no es positivo.")

    relevant_mask = Pv >= p_floor_rel * pmax
    if np.any(relevant_mask):
        U_vis_max = float(np.nanquantile(Uv[relevant_mask], relevant_quantile))
    else:
        U_vis_max = float(np.nanquantile(Uv, 0.95))

    if not np.isfinite(U_vis_max) or U_vis_max <= 0.0:
        U_vis_max = float(np.nanmax(Uv))
    if not np.isfinite(U_vis_max) or U_vis_max <= 0.0:
        U_vis_max = 1.0

    U_clip = np.minimum(Uv, U_vis_max)
    Z = -z_depth * (1.0 - U_clip / U_vis_max)

    # Mismos colores que el mapa 2D de probabilidad, pero invertidos respecto
    # del valor numérico de U: los mínimos del potencial, que corresponden a
    # máximos de probabilidad, aparecen en amarillo como en P_st.
    cmap = plt.get_cmap(plt.rcParams.get("image.cmap", "turbo")).reversed()
    norm = colors.Normalize(vmin=-z_depth, vmax=0.0)
    facecolors = cmap(norm(Z))

    # ------------------------------------------------------------
    # Figura 3D con layout manual.
    # Se evita tight_layout porque en ejes 3D suele recortar
    # etiquetas, eje z o la barra de color.
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(8.2, 6.2))

    # Eje principal 3D. Se reserva espacio explícito a la derecha
    # para la barra de color.
    ax = fig.add_axes([0.04, 0.08, 0.72, 0.84], projection="3d")

    ax.plot_surface(
        X1v,
        X2v,
        Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )

    ax.set_xlabel(r"$x_1$", labelpad=8)
    ax.set_ylabel(r"$x_2$", labelpad=8)
    ax.set_zlabel(r"$U$", labelpad=10)

    if show_title:
        ax.set_title(title)

    ax.set_xlim(*visual_xlim)
    ax.set_ylim(*visual_ylim)
    ax.set_zlim(-z_depth, 0.0)
    ax.view_init(elev=elev, azim=azim)
    ax.invert_xaxis()

    try:
        ax.set_box_aspect((1.0, 1.0, 0.65))
    except Exception:
        pass

    # ------------------------------------------------------------
    # Barra de color con eje propio.
    # La barra usa exactamente el mismo cmap y norm que la superficie.
    # ------------------------------------------------------------
    if show_colorbar:
        cax = fig.add_axes([0.82, 0.22, 0.035, 0.56])
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(Z)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(r"$U$", rotation=90, labelpad=12)

    # ------------------------------------------------------------
    # Guardado.
    # No se usa bbox_inches="tight" para evitar recortes en 3D.
    # ------------------------------------------------------------
    fig.savefig(output_path, dpi=300, facecolor="white")

    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")

    plt.close(fig)

def run_case(args: argparse.Namespace) -> None:
    """Ejecuta una simulación y guarda salidas."""
    params = ModelParams(a=args.a, b=args.b, k=args.k, S=args.S, n=args.n, D=args.D)
    grid = GridParams(xmin=args.xmin, xmax=args.xmax, N=args.N)
    solver = SolverParams(
        safety=args.safety,
        max_steps=args.max_steps,
        check_every=args.check_every,
        tol=args.tol,
        sigma=args.sigma,
        init_mode=args.init_mode,
        clip_negative=not args.no_clip_negative,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    P, X1, X2, diagnostics = evolve_fokker_planck(params, grid, solver)
    U = effective_potential(P)

    tag = a_tag(params.a)
    prefix = "test_" if args.test_prefix else ""

    prob_path = output_dir / f"{prefix}fp_prob_{tag}_N{grid.N}.png"
    pot_path = output_dir / f"{prefix}fp_potential_{tag}_N{grid.N}.png"
    iso_path = output_dir / f"{prefix}fp_potential_isometric_{tag}_N{grid.N}.png"
    diag_path = output_dir / f"{prefix}fp_diagnostics_{tag}_N{grid.N}.json"
    data_path = output_dir / f"{prefix}fp_data_{tag}_N{grid.N}.npz"

    if args.save_figures:
        plot_density(P, X1, X2, rf"$P_{{st}}(x_1,x_2)$, $a={params.a:.2f}$", r"$P_{st}$", prob_path)

        # Para representar U se recorta el extremo superior visual, que suele estar dominado por la frontera.
        U_plot = np.minimum(U, np.quantile(U, 0.99))
        plot_density(U_plot, X1, X2, rf"$U(x_1,x_2)$, $a={params.a:.2f}$", r"$U=-\ln(P+\delta)$", pot_path)
        plot_potential_isometric(
            U,
            P,
            X1,
            X2,
            rf"$U(x_1,x_2)$, $a={params.a:.2f}$",
            iso_path,
        )

    np.savez_compressed(data_path, P=P, U=U, X1=X1, X2=X2)
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    final = diagnostics["final"]
    print("Simulación Fokker--Planck completada")
    print(f"  a = {params.a}")
    print(f"  N = {grid.N}, puntos por eje = {grid.N + 1}")
    print(f"  dt = {diagnostics['time_step']['dt']:.6e}")
    print(f"  pasos = {final['steps']}, convergió = {final['converged']}")
    print(f"  delta_max final = {final['delta_max']:.3e}")
    print(f"  masa final = {final['mass']:.12f}")
    print(f"  min(P) = {final['min_P']:.3e}, max(P) = {final['max_P']:.3e}")
    print(f"  máximo de P en (x1,x2)=({final['argmax_x1']:.4f}, {final['argmax_x2']:.4f})")
    print(f"  diagnósticos: {diag_path}")
    print(f"  datos: {data_path}")
    if args.save_figures:
        print(f"  figura P: {prob_path}")
        print(f"  figura U: {pot_path}")
        print(f"  figura U isométrica: {iso_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fokker--Planck para el paisaje de Waddington.")
    parser.add_argument("--a", type=float, default=1.48, help="Parámetro de autoactivación.")
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--S", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--D", type=float, default=0.05)

    parser.add_argument("--xmin", type=float, default=-0.25)
    parser.add_argument("--xmax", type=float, default=3.0)
    parser.add_argument("--N", type=int, default=150, help="Número de intervalos por eje.")

    parser.add_argument("--safety", type=float, default=0.20)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument("--sigma", type=float, default=0.16)
    parser.add_argument(
        "--init-mode",
        choices=["central", "stable_fixed_points", "symmetric_mixture"],
        default="central",
        help="Condición inicial.",
    )
    parser.add_argument("--no-clip-negative", action="store_true", help="No recortar valores negativos de P.")
    parser.add_argument("--save-figures", action="store_true", help="Guardar mapas 2D de P y U.")
    parser.add_argument("--test-prefix", action="store_true", help="Añadir prefijo test_ a los archivos.")
    parser.add_argument("--output-dir", type=str, default="Imágenes")

    return parser.parse_args()


if __name__ == "__main__":
    run_case(parse_args())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tarea 3: cortes del paisaje de probabilidad de Fokker--Planck y construcción
 del paisaje dinámico de Waddington.

Este script NO modifica `fp_waddington11.py`. Lo usa como motor numérico para:

1. Resolver la ecuación de Fokker--Planck para una familia de valores de a.
2. Identificar, para cada a, el estado no diferenciado y los dos estados diferenciados.
3. Cortar P_st(x1,x2;a) a lo largo de las rectas que conectan el estado no diferenciado
   con los estados diferenciados.
4. Construir una coordenada local centrada en el estado no diferenciado.
   Puede ser física (xi, distancia en el plano) o normalizada (eta, con eta=±1
   en los estados diferenciados).
5. Apilar los cortes y construir U_slice(x,a) = -log(P_slice(x,a)+eps).
6. Guardar figuras de diagnóstico y el paisaje dinámico final.

Salida principal:
    task3_data_NXXX.npz
    task3_summary_NXXX.json
    task3_waddington_heatmap_NXXX.png/pdf
    task3_waddington_surface_NXXX.png/pdf
    task3_slice_diagnostics_NXXX.png/pdf

Ejemplo rápido:
    python task3_waddington_slices.py --num-a 7 --N 80 --max-steps 3000

Ejemplo más serio:
    python task3_waddington_slices.py --num-a 60 --N 150 --max-steps 40000 \
        --tol 1e-6 --init-mode symmetric_mixture --coordinate-mode normalized \
        --output-dir Resultados_task3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import brentq
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Este script necesita scipy. Instala scipy o ejecútalo en el mismo entorno "
        "en el que ejecutas fp_waddington11.py."
    ) from exc

# -----------------------------------------------------------------------------
# Importación robusta del script de la Tarea 2.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fp_waddington11 import (  # noqa: E402
    GridParams,
    ModelParams,
    SolverParams,
    evolve_fokker_planck,
    find_fixed_points,
    hill_field,
)


# =============================================================================
# Utilidades generales
# =============================================================================


def parse_float_list(text: str) -> np.ndarray:
    """Convierte una cadena tipo '0.1,0.5,1.0' en un array ordenado."""
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("La lista de valores de a está vacía.")
    return np.array(sorted(values), dtype=float)


def json_safe(obj):
    """Convierte objetos numpy a estructuras serializables en JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def grid_axes(X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae los ejes unidimensionales de una malla generada con indexing='ij'."""
    return X1[:, 0], X2[0, :]


def point_from_dict(p: Dict[str, object]) -> np.ndarray:
    """Convierte un punto fijo guardado como diccionario a vector numpy."""
    return np.array([float(p["x1"]), float(p["x2"])], dtype=float)


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Normaliza un vector 2D y comprueba que no sea degenerado."""
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise RuntimeError("No se puede construir un vector unitario a partir de un vector degenerado.")
    return vec / norm


# =============================================================================
# Identificación geométrica de estados: no diferenciado y diferenciados
# =============================================================================


def diagonal_field_value(x: float, params: ModelParams) -> float:
    """
    Campo de Hill sobre la diagonal x1=x2=x.

    En la diagonal, por simetría F1=F2. Las raíces de esta función son candidatos
    al estado no diferenciado, incluso cuando este estado pierde estabilidad.
    """
    F1, _ = hill_field(np.array(x), np.array(x), params)
    return float(F1)


def find_diagonal_fixed_point(params: ModelParams, grid: GridParams, n_scan: int = 1000) -> np.ndarray:
    """
    Encuentra un punto fijo sobre la diagonal x1=x2.

    Se buscan cambios de signo de F(x,x) en el dominio. Si hay más de una raíz
    diagonal, se escoge la que queda más cerca de a en distancia absoluta, que en
    este modelo suele corresponder a la rama central usada como referencia.
    """
    xs = np.linspace(grid.xmin, grid.xmax, n_scan)
    vals = np.array([diagonal_field_value(float(x), params) for x in xs])

    roots: List[float] = []
    for i in range(len(xs) - 1):
        f0, f1 = vals[i], vals[i + 1]
        if not np.isfinite(f0) or not np.isfinite(f1):
            continue
        if f0 == 0.0:
            roots.append(float(xs[i]))
        elif f0 * f1 < 0.0:
            try:
                roots.append(float(brentq(lambda z: diagonal_field_value(z, params), xs[i], xs[i + 1])))
            except ValueError:
                continue

    # Quitar duplicados numéricos.
    unique_roots: List[float] = []
    for r in roots:
        if all(abs(r - q) > 1.0e-7 for q in unique_roots):
            unique_roots.append(r)

    if not unique_roots:
        # Fallback: usar el punto de la diagonal donde |F| es mínimo.
        idx = int(np.nanargmin(np.abs(vals)))
        r = float(xs[idx])
    else:
        target = np.clip(params.a, grid.xmin, grid.xmax)
        r = min(unique_roots, key=lambda z: abs(z - target))

    return np.array([r, r], dtype=float)


def identify_reference_states(
    fixed_points: Sequence[Dict[str, object]],
    params: ModelParams,
    grid: GridParams,
    require_stable_differentiated: bool = True,
) -> Dict[str, object]:
    """
    Identifica los tres puntos usados en el corte.

    Devuelve:
        x_nd      : estado no diferenciado sobre x1=x2.
        x_d_plus  : estado diferenciado con x1>x2.
        x_d_minus : estado diferenciado con x2>x1.

    El estado no diferenciado se calcula de forma independiente sobre la diagonal,
    porque para a<a_c puede ser inestable y aun así debe usarse como centro
    geométrico del corte. Para los diferenciados se prefieren puntos fijos estables.
    """
    x_nd = find_diagonal_fixed_point(params, grid)

    points = list(fixed_points)
    if not points:
        raise RuntimeError("No se encontraron puntos fijos; no se pueden definir los cortes.")

    if require_stable_differentiated:
        candidates = [p for p in points if str(p.get("type", "")) == "stable"]
        # Cerca de la bifurcación o si falla la clasificación puede no haber suficientes.
        if len(candidates) < 2:
            candidates = points
    else:
        candidates = points

    plus = [p for p in candidates if float(p["x1"]) - float(p["x2"]) > 1.0e-5]
    minus = [p for p in candidates if float(p["x2"]) - float(p["x1"]) > 1.0e-5]

    if not plus or not minus:
        # Fallback: usar todos los puntos fijos y maximizar la asimetría.
        plus = [p for p in points if float(p["x1"]) - float(p["x2"]) > 1.0e-5]
        minus = [p for p in points if float(p["x2"]) - float(p["x1"]) > 1.0e-5]

    if not plus or not minus:
        raise RuntimeError(
            "No se han podido identificar los dos estados diferenciados. "
            "Revisa las semillas de find_fixed_points o el rango de a."
        )

    # Selección: punto más alejado de la diagonal, equivalente al destino más diferenciado.
    p_plus = max(plus, key=lambda p: float(p["x1"]) - float(p["x2"]))
    p_minus = max(minus, key=lambda p: float(p["x2"]) - float(p["x1"]))

    x_d_plus = point_from_dict(p_plus)
    x_d_minus = point_from_dict(p_minus)

    return {
        "x_nd": x_nd,
        "x_d_plus": x_d_plus,
        "x_d_minus": x_d_minus,
        "d_plus": float(np.linalg.norm(x_d_plus - x_nd)),
        "d_minus": float(np.linalg.norm(x_d_minus - x_nd)),
        "plus_type": str(p_plus.get("type", "unknown")),
        "minus_type": str(p_minus.get("type", "unknown")),
    }


# =============================================================================
# Construcción de cortes
# =============================================================================


def build_cut_points(
    local_values: np.ndarray,
    x_nd: np.ndarray,
    x_d_plus: np.ndarray,
    x_d_minus: np.ndarray,
    coordinate_mode: str = "normalized",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye los puntos del corte en el plano (x1,x2).

    Modos disponibles:

    normalized:
        local = eta. Se define eta=0 en el estado no diferenciado,
        eta=+1 en el estado diferenciado con x1>x2 y eta=-1 en el estado
        diferenciado con x2>x1. Es el modo recomendado para la figura final,
        porque todos los cortes comparan la región entre los estados relevantes
        sin prolongar artificialmente la recta fuera del dominio.

    physical:
        local = xi. Se mide distancia física en el plano (x1,x2) desde el estado
        no diferenciado. Este modo conserva longitudes reales, pero puede hacer
        que parte del corte salga del dominio si se usa una extensión común para
        todos los valores de a.
    """
    if coordinate_mode not in {"normalized", "physical"}:
        raise ValueError("coordinate_mode debe ser 'normalized' o 'physical'.")

    v_plus = x_d_plus - x_nd
    v_minus = x_d_minus - x_nd
    e_plus = unit_vector(v_plus)
    e_minus = unit_vector(v_minus)

    points = np.zeros((len(local_values), 2), dtype=float)
    for i, value in enumerate(local_values):
        if value >= 0.0:
            if coordinate_mode == "normalized":
                points[i] = x_nd + value * v_plus
            else:
                points[i] = x_nd + value * e_plus
        else:
            if coordinate_mode == "normalized":
                points[i] = x_nd + abs(value) * v_minus
            else:
                points[i] = x_nd + abs(value) * e_minus

    return points, e_plus, e_minus


def interpolate_probability_on_cut(
    P: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    cut_points: np.ndarray,
) -> np.ndarray:
    """
    Interpola la densidad P sobre los puntos del corte.

    Se interpola P y no U para evitar que el logaritmo amplifique artefactos en
    zonas de muy baja probabilidad. Los puntos fuera del dominio se devuelven como NaN.
    """
    x1_axis, x2_axis = grid_axes(X1, X2)
    interp = RegularGridInterpolator(
        (x1_axis, x2_axis),
        P,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    P_cut = interp(cut_points)
    # Pequeños valores negativos pueden aparecer por interpolación si P tiene ruido numérico.
    return np.where(P_cut < 0.0, 0.0, P_cut)


def compute_global_potential_from_slices(P_slices: np.ndarray, eps_rel: float) -> np.ndarray:
    """
    Calcula U_slice=-log(P_slice+eps) usando un eps global.

    El desplazamiento del mínimo también es global. Así se conserva la comparación
    relativa entre distintos valores de a mejor que si se normalizara cada corte
    por separado.
    """
    pmax = float(np.nanmax(P_slices))
    if not np.isfinite(pmax) or pmax <= 0.0:
        raise RuntimeError("No se puede calcular U_slice porque max(P_slice) no es positivo.")
    eps = eps_rel * pmax
    U = -np.log(P_slices + eps)
    U = U - np.nanmin(U)
    return U


# =============================================================================
# Simulación y apilamiento
# =============================================================================


def run_single_a(
    a: float,
    model_base: ModelParams,
    grid: GridParams,
    solver: SolverParams,
    require_stable_differentiated: bool,
) -> Dict[str, object]:
    """Ejecuta Fokker--Planck para un valor de a e identifica los estados de referencia."""
    params = ModelParams(
        a=float(a),
        b=model_base.b,
        k=model_base.k,
        S=model_base.S,
        n=model_base.n,
        D=model_base.D,
    )

    P, X1, X2, diagnostics = evolve_fokker_planck(params, grid, solver)
    fixed_points = diagnostics.get("fixed_points", find_fixed_points(params, grid))
    states = identify_reference_states(
        fixed_points=fixed_points,
        params=params,
        grid=grid,
        require_stable_differentiated=require_stable_differentiated,
    )

    return {
        "a": float(a),
        "params": params,
        "P": P,
        "X1": X1,
        "X2": X2,
        "diagnostics": diagnostics,
        "fixed_points": fixed_points,
        "states": states,
    }


def choose_local_values(
    runs: Sequence[Dict[str, object]],
    n_xi: int,
    xi_max: Optional[float],
    xi_margin: float,
    coordinate_mode: str,
) -> np.ndarray:
    """Define una malla común de coordenada local para todos los valores de a."""
    if coordinate_mode not in {"normalized", "physical"}:
        raise ValueError("coordinate_mode debe ser 'normalized' o 'physical'.")

    if xi_max is not None:
        local_max = float(xi_max)
    elif coordinate_mode == "normalized":
        # En coordenada normalizada, eta=±1 coincide con los estados diferenciados.
        # No añadimos margen por defecto para no prolongar el corte fuera del dominio.
        local_max = 1.0
    else:
        max_distance = 0.0
        for run in runs:
            st = run["states"]
            max_distance = max(max_distance, float(st["d_plus"]), float(st["d_minus"]))
        local_max = max_distance + xi_margin

    if not np.isfinite(local_max) or local_max <= 0.0:
        raise RuntimeError("La extensión de la coordenada local no es válida.")
    return np.linspace(-local_max, local_max, n_xi)


def build_stacked_slices(
    runs: Sequence[Dict[str, object]],
    xi_values: np.ndarray,
    eps_rel: float,
    coordinate_mode: str,
) -> Dict[str, object]:
    """Construye P_slice y U_slice para toda la familia de valores de a."""
    P_slices: List[np.ndarray] = []
    cut_points_all: List[np.ndarray] = []
    e_plus_all: List[np.ndarray] = []
    e_minus_all: List[np.ndarray] = []

    for run in runs:
        st = run["states"]
        points, e_plus, e_minus = build_cut_points(
            local_values=xi_values,
            x_nd=np.asarray(st["x_nd"], dtype=float),
            x_d_plus=np.asarray(st["x_d_plus"], dtype=float),
            x_d_minus=np.asarray(st["x_d_minus"], dtype=float),
            coordinate_mode=coordinate_mode,
        )
        P_cut = interpolate_probability_on_cut(
            P=np.asarray(run["P"], dtype=float),
            X1=np.asarray(run["X1"], dtype=float),
            X2=np.asarray(run["X2"], dtype=float),
            cut_points=points,
        )
        P_slices.append(P_cut)
        cut_points_all.append(points)
        e_plus_all.append(e_plus)
        e_minus_all.append(e_minus)

    P_stack = np.vstack(P_slices)
    U_stack = compute_global_potential_from_slices(P_stack, eps_rel=eps_rel)

    return {
        "P_slices": P_stack,
        "U_slices": U_stack,
        "cut_points": np.stack(cut_points_all),
        "e_plus": np.stack(e_plus_all),
        "e_minus": np.stack(e_minus_all),
    }


# =============================================================================
# Figuras
# =============================================================================


def clipped_for_visualization(Z: np.ndarray, q: float = 0.985) -> np.ndarray:
    """Recorte visual por cuantiles para que las zonas P≈0 no dominen la escala."""
    zmax = float(np.nanquantile(Z, q))
    if not np.isfinite(zmax):
        return Z
    return np.minimum(Z, zmax)


def local_coordinate_label(coordinate_mode: str) -> str:
    """Etiqueta LaTeX para la coordenada local usada en las figuras."""
    if coordinate_mode == "normalized":
        return r"Coordenada local normalizada $\eta$"
    return r"Coordenada local $\xi$"


def local_coordinate_symbol(coordinate_mode: str) -> str:
    """Símbolo LaTeX corto de la coordenada local."""
    if coordinate_mode == "normalized":
        return r"$\eta$"
    return r"$\xi$"


def plot_waddington_heatmap(
    xi_values: np.ndarray,
    a_values: np.ndarray,
    U_slices: np.ndarray,
    output_path: Path,
    critical_a: Optional[float] = None,
    coordinate_mode: str = "normalized",
) -> None:
    """Guarda una vista cenital de U_slice(xi,a)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    U_plot = clipped_for_visualization(U_slices)
    im = ax.pcolormesh(xi_values, a_values, U_plot, shading="auto")
    ax.set_xlabel(local_coordinate_label(coordinate_mode))
    ax.set_ylabel(r"Parámetro de autoactivación $a$")
    symbol = r"\eta" if coordinate_mode == "normalized" else r"\xi"
    ax.set_title(rf"Paisaje dinámico de Waddington: $U_{{\mathrm{{slice}}}}({symbol},a)$")
    if critical_a is not None:
        ax.axhline(float(critical_a), color="k", linestyle="--", linewidth=1.0, label=rf"$a_c={critical_a:.4f}$")
        ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label=r"$U_{\mathrm{slice}}=-\ln(P_{\mathrm{slice}}+\varepsilon)$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_waddington_surface(
    xi_values: np.ndarray,
    a_values: np.ndarray,
    U_slices: np.ndarray,
    output_path: Path,
    critical_a: Optional[float] = None,
    coordinate_mode: str = "normalized",
) -> None:
    """Guarda una superficie 3D de U_slice(xi,a)."""
    Xi, A = np.meshgrid(xi_values, a_values)
    Z = clipped_for_visualization(U_slices)

    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xi, A, Z, cmap="turbo", linewidth=0.0, antialiased=True)
    ax.set_xlabel(local_coordinate_symbol(coordinate_mode))
    ax.set_ylabel(r"$a$")
    ax.set_zlabel(r"$U_{\mathrm{slice}}$")
    ax.set_title(r"Paisaje dinámico de Waddington")
    ax.view_init(elev=25, azim=-55)
    try:
        ax.set_box_aspect((1.4, 1.0, 0.7))
    except Exception:
        pass
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.12, label=r"$U_{\mathrm{slice}}$")

    # Línea auxiliar en el plano a=a_c sobre la pared posterior, solo como referencia visual.
    if critical_a is not None and np.min(a_values) <= critical_a <= np.max(a_values):
        ax.plot(
            [float(np.min(xi_values)), float(np.max(xi_values))],
            [float(critical_a), float(critical_a)],
            [float(np.nanmax(Z)), float(np.nanmax(Z))],
            color="k",
            linestyle="--",
            linewidth=1.0,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def select_nearest_runs(
    runs: Sequence[Dict[str, object]],
    requested_a_values: Sequence[float],
) -> List[int]:
    """Devuelve índices de las simulaciones más cercanas a ciertos valores de a."""
    a_values = np.array([float(run["a"]) for run in runs])
    idxs: List[int] = []
    for a0 in requested_a_values:
        idx = int(np.argmin(np.abs(a_values - float(a0))))
        if idx not in idxs:
            idxs.append(idx)
    return idxs


def plot_slice_diagnostics(
    runs: Sequence[Dict[str, object]],
    xi_values: np.ndarray,
    U_slices: np.ndarray,
    cut_points: np.ndarray,
    output_path: Path,
    requested_a_values: Sequence[float],
    coordinate_mode: str = "normalized",
) -> None:
    """
    Figura de diagnóstico: paisajes P(x1,x2) con corte superpuesto y curvas U_slice.
    """
    idxs = select_nearest_runs(runs, requested_a_values)
    ncols = len(idxs)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 7.0), squeeze=False)

    for col, idx in enumerate(idxs):
        run = runs[idx]
        a = float(run["a"])
        P = np.asarray(run["P"])
        X1 = np.asarray(run["X1"])
        X2 = np.asarray(run["X2"])
        pts = cut_points[idx]
        st = run["states"]

        ax = axes[0, col]
        im = ax.pcolormesh(X1, X2, P, shading="auto")
        ax.plot(pts[:, 0], pts[:, 1], "w-", linewidth=1.5, label="corte")
        ax.scatter([st["x_nd"][0]], [st["x_nd"][1]], c="white", edgecolors="black", s=55, marker="o", label="ND")
        ax.scatter(
            [st["x_d_plus"][0], st["x_d_minus"][0]],
            [st["x_d_plus"][1], st["x_d_minus"][1]],
            c="red",
            edgecolors="black",
            s=55,
            marker="o",
            label="D",
        )
        ax.set_title(rf"$P_{{st}}(x_1,x_2)$, $a={a:.3f}$")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.legend(loc="upper right", fontsize=8)

        ax2 = axes[1, col]
        ax2.plot(xi_values, U_slices[idx], linewidth=1.6)
        ax2.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax2.set_xlabel(local_coordinate_symbol(coordinate_mode))
        ax2.set_ylabel(r"$U_{\mathrm{slice}}$")
        ax2.set_title(rf"Corte local, $a={a:.3f}$")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


# =============================================================================
# Validación y guardado
# =============================================================================


def build_summary(
    runs: Sequence[Dict[str, object]],
    xi_values: np.ndarray,
    stacked: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Crea un resumen de diagnóstico de la simulación completa."""
    run_summaries = []
    for i, run in enumerate(runs):
        final = run["diagnostics"].get("final", {})
        P_cut = stacked["P_slices"][i]
        U_cut = stacked["U_slices"][i]
        run_summaries.append(
            {
                "a": float(run["a"]),
                "converged": bool(final.get("converged", False)),
                "steps": int(final.get("steps", -1)),
                "delta_max": float(final.get("delta_max", np.nan)),
                "mass": float(final.get("mass", np.nan)),
                "min_P": float(final.get("min_P", np.nan)),
                "min_before_clip": float(final.get("min_before_clip", np.nan)),
                "max_P": float(final.get("max_P", np.nan)),
                "argmax_x1": float(final.get("argmax_x1", np.nan)),
                "argmax_x2": float(final.get("argmax_x2", np.nan)),
                "x_nd": np.asarray(run["states"]["x_nd"]).tolist(),
                "x_d_plus": np.asarray(run["states"]["x_d_plus"]).tolist(),
                "x_d_minus": np.asarray(run["states"]["x_d_minus"]).tolist(),
                "d_plus": float(run["states"]["d_plus"]),
                "d_minus": float(run["states"]["d_minus"]),
                "plus_type": run["states"]["plus_type"],
                "minus_type": run["states"]["minus_type"],
                "nan_fraction_cut": float(np.mean(~np.isfinite(P_cut))),
                "min_U_cut_local_coordinate": float(xi_values[int(np.nanargmin(U_cut))]),
                "min_U_cut": float(np.nanmin(U_cut)),
            }
        )

    warnings = []
    for r in run_summaries:
        if not r["converged"]:
            warnings.append(f"a={r['a']:.4f}: la simulación no alcanzó el criterio de convergencia.")
        if abs(r["mass"] - 1.0) > 1.0e-8:
            warnings.append(f"a={r['a']:.4f}: masa final distinta de 1 dentro de tolerancia: {r['mass']:.12f}.")
        if r["min_before_clip"] < -1.0e-8:
            warnings.append(f"a={r['a']:.4f}: aparecen valores negativos antes del clip: {r['min_before_clip']:.3e}.")
        if r["nan_fraction_cut"] > 0.0:
            warnings.append(f"a={r['a']:.4f}: parte del corte cae fuera del dominio: {r['nan_fraction_cut']:.3%}.")

    return {
        "description": "Tarea 3: cortes de P_st y paisaje dinamico de Waddington.",
        "args": vars(args),
        "coordinate_mode": str(args.coordinate_mode),
        "local_coordinate_min": float(np.min(xi_values)),
        "local_coordinate_max": float(np.max(xi_values)),
        "n_xi": int(len(xi_values)),
        "n_a": int(len(runs)),
        "runs": run_summaries,
        "warnings": warnings,
    }


def save_outputs(
    output_dir: Path,
    grid: GridParams,
    a_values: np.ndarray,
    xi_values: np.ndarray,
    runs: Sequence[Dict[str, object]],
    stacked: Dict[str, object],
    summary: Dict[str, object],
) -> Tuple[Path, Path]:
    """Guarda arrays principales y resumen JSON."""
    data_path = output_dir / f"task3_data_N{grid.N}.npz"
    json_path = output_dir / f"task3_summary_N{grid.N}.json"

    np.savez_compressed(
        data_path,
        a_values=a_values,
        xi_values=xi_values,
        P_slices=stacked["P_slices"],
        U_slices=stacked["U_slices"],
        cut_points=stacked["cut_points"],
        e_plus=stacked["e_plus"],
        e_minus=stacked["e_minus"],
        x_nd=np.array([run["states"]["x_nd"] for run in runs]),
        x_d_plus=np.array([run["states"]["x_d_plus"] for run in runs]),
        x_d_minus=np.array([run["states"]["x_d_minus"] for run in runs]),
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)

    return data_path, json_path


# =============================================================================
# CLI principal
# =============================================================================


def build_a_values(args: argparse.Namespace) -> np.ndarray:
    """Construye la lista de valores de a del barrido."""
    if args.a_values:
        return parse_float_list(args.a_values)
    return np.linspace(args.a_min, args.a_max, args.num_a)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tarea 3: cortes del paisaje de Waddington.")

    # Modelo
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--S", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--D", type=float, default=0.05)

    # Malla espacial
    parser.add_argument("--xmin", type=float, default=-0.25)
    parser.add_argument("--xmax", type=float, default=3.0)
    parser.add_argument("--N", type=int, default=120, help="Número de intervalos por eje.")

    # Solver Fokker-Planck
    parser.add_argument("--safety", type=float, default=0.20)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument("--sigma", type=float, default=0.16)
    parser.add_argument(
        "--init-mode",
        choices=["central", "stable_fixed_points", "symmetric_mixture"],
        default="symmetric_mixture",
        help="Condición inicial usada en cada simulación de Fokker-Planck.",
    )
    parser.add_argument("--no-clip-negative", action="store_true")

    # Barrido en a
    parser.add_argument("--a-min", type=float, default=0.05)
    parser.add_argument("--a-max", type=float, default=1.80)
    parser.add_argument("--num-a", type=int, default=35)
    parser.add_argument(
        "--a-values",
        type=str,
        default="",
        help="Lista explícita separada por comas. Si se usa, ignora a-min/a-max/num-a.",
    )
    parser.add_argument("--critical-a", type=float, default=0.78481, help="Valor crítico usado solo como referencia visual.")

    # Coordenada local
    parser.add_argument(
        "--coordinate-mode",
        choices=["normalized", "physical"],
        default="normalized",
        help=(
            "Modo de coordenada local. 'normalized' usa eta=0 en ND y eta=±1 en los "
            "estados diferenciados. 'physical' usa distancia real xi en el plano (x1,x2)."
        ),
    )
    parser.add_argument("--n-xi", type=int, default=401)
    parser.add_argument("--xi-max", type=float, default=None, help="Extensión máxima de |xi|. Si se omite, se calcula automáticamente.")
    parser.add_argument("--xi-margin", type=float, default=0.20)
    parser.add_argument("--eps-rel", type=float, default=1.0e-12)
    parser.add_argument(
        "--allow-unstable-differentiated",
        action="store_true",
        help="No exigir que los estados diferenciados elegidos estén clasificados como estables.",
    )

    # Figuras y salida
    parser.add_argument("--output-dir", type=str, default="Resultados_task3")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--diagnostic-a-values", type=str, default="0.1,1.05,1.48")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_base = ModelParams(a=0.0, b=args.b, k=args.k, S=args.S, n=args.n, D=args.D)
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

    a_values = build_a_values(args)
    if len(a_values) < 2:
        raise ValueError("Se necesitan al menos dos valores de a para construir un paisaje dinámico.")

    print("Tarea 3: construcción del paisaje dinámico de Waddington")
    print(f"  Valores de a: {len(a_values)} desde {a_values[0]:.4f} hasta {a_values[-1]:.4f}")
    print(f"  Malla espacial: N={grid.N}, dominio=[{grid.xmin}, {grid.xmax}]^2")
    print(f"  init_mode={solver.init_mode}, max_steps={solver.max_steps}, tol={solver.tol:.1e}")
    print(f"  coordinate_mode={args.coordinate_mode}")

    runs: List[Dict[str, object]] = []
    for i, a in enumerate(a_values, start=1):
        print(f"\n[{i}/{len(a_values)}] Ejecutando a={a:.5f}")
        run = run_single_a(
            a=float(a),
            model_base=model_base,
            grid=grid,
            solver=solver,
            require_stable_differentiated=not args.allow_unstable_differentiated,
        )
        final = run["diagnostics"].get("final", {})
        print(
            "  converged={conv}, steps={steps}, delta={delta:.3e}, mass={mass:.12f}".format(
                conv=final.get("converged", False),
                steps=final.get("steps", -1),
                delta=float(final.get("delta_max", np.nan)),
                mass=float(final.get("mass", np.nan)),
            )
        )
        st = run["states"]
        print(
            "  ND=({:.4f},{:.4f}), D+=({:.4f},{:.4f}), D-=({:.4f},{:.4f})".format(
                st["x_nd"][0],
                st["x_nd"][1],
                st["x_d_plus"][0],
                st["x_d_plus"][1],
                st["x_d_minus"][0],
                st["x_d_minus"][1],
            )
        )
        runs.append(run)

    xi_values = choose_local_values(
        runs=runs,
        n_xi=args.n_xi,
        xi_max=args.xi_max,
        xi_margin=args.xi_margin,
        coordinate_mode=args.coordinate_mode,
    )
    stacked = build_stacked_slices(
        runs=runs,
        xi_values=xi_values,
        eps_rel=args.eps_rel,
        coordinate_mode=args.coordinate_mode,
    )
    summary = build_summary(runs=runs, xi_values=xi_values, stacked=stacked, args=args)

    data_path, json_path = save_outputs(
        output_dir=output_dir,
        grid=grid,
        a_values=a_values,
        xi_values=xi_values,
        runs=runs,
        stacked=stacked,
        summary=summary,
    )

    if not args.no_figures:
        heatmap_path = output_dir / f"task3_waddington_heatmap_N{grid.N}.png"
        surface_path = output_dir / f"task3_waddington_surface_N{grid.N}.png"
        diagnostic_path = output_dir / f"task3_slice_diagnostics_N{grid.N}.png"

        plot_waddington_heatmap(
            xi_values=xi_values,
            a_values=a_values,
            U_slices=stacked["U_slices"],
            output_path=heatmap_path,
            critical_a=args.critical_a,
            coordinate_mode=args.coordinate_mode,
        )
        plot_waddington_surface(
            xi_values=xi_values,
            a_values=a_values,
            U_slices=stacked["U_slices"],
            output_path=surface_path,
            critical_a=args.critical_a,
            coordinate_mode=args.coordinate_mode,
        )
        diagnostic_values = parse_float_list(args.diagnostic_a_values)
        plot_slice_diagnostics(
            runs=runs,
            xi_values=xi_values,
            U_slices=stacked["U_slices"],
            cut_points=stacked["cut_points"],
            output_path=diagnostic_path,
            requested_a_values=diagnostic_values,
            coordinate_mode=args.coordinate_mode,
        )

    print("\nProceso completado.")
    print(f"  Datos: {data_path}")
    print(f"  Diagnóstico JSON: {json_path}")
    if summary["warnings"]:
        print("\nAdvertencias:")
        for w in summary["warnings"]:
            print(f"  - {w}")
    else:
        print("  Sin advertencias de validación básicas.")


if __name__ == "__main__":
    main()

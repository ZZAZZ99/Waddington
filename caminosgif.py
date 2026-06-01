import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import warnings

# Silenciamos warnings numéricos para una salida limpia en consola
warnings.filterwarnings("ignore")

# ============================================================
# 1. PARÁMETROS DEL SISTEMA 
# ============================================================
S = 0.5
n = 4
b = 1.0  
k = 1.0  
D_scale = 0.05  

# Malla espacial: [-0.25, 3.0]
N = 150  
x_min, x_max = -0.25, 3.0
h = (x_max - x_min) / (N - 1)

x_lin = np.linspace(x_min, x_max, N)
X, Y = np.meshgrid(x_lin, x_lin)

# ============================================================
# 2. DEFINICIÓN DEL CAMPO VECTORIAL Y PUNTOS FIJOS
# ============================================================
def f(x, y, a):
    return a * (x**n) / (S**n + x**n) + b * (S**n) / (S**n + y**n) - k * x

def g(x, y, a):
    return a * (y**n) / (S**n + y**n) + b * (S**n) / (S**n + x**n) - k * y

def jacobian(x, y, a):
    dfdx = a * (n * x**(n-1) * S**n) / (S**n + x**n)**2 - k
    dfdy = -b * (n * y**(n-1) * S**n) / (S**n + y**n)**2
    dgdx = -b * (n * x**(n-1) * S**n) / (S**n + x**n)**2
    dgdy = a * (n * y**(n-1) * S**n) / (S**n + y**n)**2 - k
    return np.array([[dfdx, dfdy], [dgdx, dgdy]])

def find_stable_fixed_points(a):
    # SOLUCIÓN AL SALTO: Malla densa sistemática de 25 semillas.
    # Esto garantiza que el algoritmo numérico NUNCA pierda de vista 
    # los atractores laterales, por muy arrinconados que estén al principio.
    guesses = [[x, y] for x in np.linspace(0.0, 3.0, 5) for y in np.linspace(0.0, 3.0, 5)]
    
    roots = []
    for guess in guesses:
        root, info, ier, mesg = fsolve(lambda vars: [f(vars[0], vars[1], a), g(vars[0], vars[1], a)], guess, full_output=True)
        if ier == 1 and root[0] >= -0.25 and root[1] >= -0.25:
            is_new = True
            for r in roots:
                if np.linalg.norm(root - r) < 1e-3:
                    is_new = False
                    break
            if is_new:
                J = jacobian(root[0], root[1], a)
                if np.trace(J) < 0 and np.linalg.det(J) > 0:
                    roots.append(root)
    return roots

# ============================================================
# 3. CONSTRUCCIÓN Y MINIMIZACIÓN DE LA ACCIÓN (GRAFO ISÓTROPO)
# ============================================================
def compute_quasipotential_landscape(a):
    U_vec = f(X, Y, a)
    V_vec = g(X, Y, a)
    M_vec = np.sqrt(U_vec**2 + V_vec**2)
    
    node_ids = np.arange(N * N).reshape((N, N))
    
    # Conexiones Ortogonales
    w_right = h * (M_vec[:, :-1] - U_vec[:, :-1]).flatten()
    w_left  = h * (M_vec[:, 1:] + U_vec[:, 1:]).flatten()
    w_up    = h * (M_vec[:-1, :] - V_vec[:-1, :]).flatten()
    w_down  = h * (M_vec[1:, :] + V_vec[1:, :]).flatten()
    
    src_ortho = np.concatenate([node_ids[:, :-1].flatten(), node_ids[:, 1:].flatten(), node_ids[:-1, :].flatten(), node_ids[1:, :].flatten()])
    tgt_ortho = np.concatenate([node_ids[:, 1:].flatten(), node_ids[:, :-1].flatten(), node_ids[1:, :].flatten(), node_ids[:-1, :].flatten()])
    w_ortho = np.concatenate([w_right, w_left, w_up, w_down])

    # Conexiones Diagonales
    sqrt2 = np.sqrt(2)
    w_ur = h * (sqrt2 * M_vec[:-1, :-1] - (U_vec[:-1, :-1] + V_vec[:-1, :-1])).flatten()
    w_dl = h * (sqrt2 * M_vec[1:, 1:] - (-U_vec[1:, 1:] - V_vec[1:, 1:])).flatten()
    w_dr = h * (sqrt2 * M_vec[1:, :-1] - (U_vec[1:, :-1] - V_vec[1:, :-1])).flatten()
    w_ul = h * (sqrt2 * M_vec[:-1, 1:] - (-U_vec[:-1, 1:] + V_vec[:-1, 1:])).flatten()

    src_diag = np.concatenate([node_ids[:-1, :-1].flatten(), node_ids[1:, 1:].flatten(), node_ids[1:, :-1].flatten(), node_ids[:-1, 1:].flatten()])
    tgt_diag = np.concatenate([node_ids[1:, 1:].flatten(), node_ids[:-1, :-1].flatten(), node_ids[:-1, 1:].flatten(), node_ids[1:, :-1].flatten()])
    w_diag = np.concatenate([w_ur, w_dl, w_dr, w_ul])

    # Ensamblaje
    sources = np.concatenate([src_ortho, src_diag])
    targets = np.concatenate([tgt_ortho, tgt_diag])
    weights = np.concatenate([w_ortho, w_diag])
    
    weights = np.maximum(weights, 1e-12)
    graph = csr_matrix((weights, (sources, targets)), shape=(N*N, N*N))
    
    fixed_points = find_stable_fixed_points(a)
    attractor_nodes = []
    
    for fp in fixed_points:
        j = np.argmin(np.abs(x_lin - fp[0]))
        i = np.argmin(np.abs(x_lin - fp[1]))
        attractor_nodes.append(i * N + j)
        
    if not attractor_nodes:
        return np.ones((N, N)) * 10.0
        
    dist_matrix = dijkstra(csgraph=graph, directed=True, indices=attractor_nodes)
    
    if len(attractor_nodes) > 1:
        S_action = np.min(dist_matrix, axis=0).reshape((N, N))
    else:
        S_action = dist_matrix.reshape((N, N))
        
    return S_action

# ============================================================
# 4. CONFIGURACIÓN Y GENERACIÓN DE LA ANIMACIÓN
# ============================================================
# Parámetros de tiempo y renderizado
fps_video = 8
num_frames_transition = 60
segundos_pausa = 1.0  # Tiempo que permanecerá estático al final
num_frames_pausa = int(fps_video * segundos_pausa)

# Construcción de la coreografía paramétrica
a_transition = np.linspace(1.50, 0.15, num_frames_transition)
a_pause = np.full(num_frames_pausa, 0.15)
a_frames = np.concatenate([a_transition, a_pause])
num_frames_total = len(a_frames)

fig = plt.figure(figsize=(14, 6))

ax_2d = fig.add_subplot(1, 2, 1)
ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

# Dibujo inicial (Dummy) para anclar la barra de color
S_ini = compute_quasipotential_landscape(a_frames[0])
P_ini = np.exp(-S_ini / D_scale)
P_ini = (P_ini / np.max(P_ini)) * 0.99 

im = ax_2d.pcolormesh(X, Y, P_ini, cmap='turbo', shading='auto', vmin=0, vmax=0.99)
cbar = fig.colorbar(im, ax=ax_2d, fraction=0.046, pad=0.04)
cbar.set_label('Probabilidad $P_{st}$', fontsize=12)

fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.85, wspace=0.2)

# Pequeña memoria caché para no recalcular la física durante la pausa estática
cache_paisaje = {"a": None, "S": None}

def update(frame):
    a_val = a_frames[frame]
    print(f"Calculando Frame {frame+1}/{num_frames_total} -> (a = {a_val:.3f})...")
    
    # Optimizamos: si el valor de 'a' no ha cambiado (estamos en la pausa), usamos la caché
    if cache_paisaje["a"] == a_val:
        S_action = cache_paisaje["S"]
    else:
        S_action = compute_quasipotential_landscape(a_val)
        cache_paisaje["a"] = a_val
        cache_paisaje["S"] = S_action
        
    P_landscape = np.exp(-S_action / D_scale)
    P_landscape = (P_landscape / np.max(P_landscape)) * 0.99 
    U_3d = -P_landscape
    
    ax_2d.clear()
    ax_3d.clear()
    
    ax_2d.pcolormesh(X, Y, P_landscape, cmap='turbo', shading='auto', vmin=0, vmax=0.99)
    ax_2d.set_title(f"Vista Cenital (a = {a_val:.3f})", fontsize=14)
    ax_2d.set_xlabel("Expresión Gen 1 ($x_1$)", fontsize=12)
    ax_2d.set_ylabel("Expresión Gen 2 ($x_2$)", fontsize=12)
    ax_2d.set_aspect('equal')
    ax_2d.set_xlim(x_min, x_max)
    ax_2d.set_ylim(x_min, x_max)
    
    ax_3d.plot_surface(X, Y, U_3d, cmap='turbo_r', edgecolor='none', alpha=0.9)
    ax_3d.set_title(f"Cuasipotencial Isométrico (a = {a_val:.3f})", fontsize=14)
    ax_3d.set_xlabel("$x_1$", fontsize=10, labelpad=5)
    ax_3d.set_ylabel("$x_2$", fontsize=10, labelpad=5)
    ax_3d.set_zlabel("U", fontsize=10, labelpad=5)
    ax_3d.set_zlim(-1.0, 0.0)
    ax_3d.view_init(elev=20, azim=-135)
    
    return ax_2d, ax_3d

print("Iniciando renderizado del GIF con rastreo robusto de atractores y pausa final...")
anim = FuncAnimation(fig, update, frames=num_frames_total, interval=120, blit=False)

output_filename = "paisaje_waddington_dinamico.gif"
anim.save(output_filename, writer='pillow', fps=fps_video)
print(f"¡Renderizado completo! Animación guardada como '{output_filename}'")
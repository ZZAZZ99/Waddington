import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.animation as animation
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. DEFINE AQUÍ TU SISTEMA 2D (Modelo de Hill Simétrico)
# ============================================================
# Parámetros fijos según la literatura de diferenciación celular
S = 0.5
n = 4
k = 1.0

def f(x, y, a, b):
    # Ecuación para el gen 1 (x)
    return a * (x**n) / (S**n + x**n) + b * (S**n) / (S**n + y**n) - k * x

def g(x, y, a, b):
    # Ecuación para el gen 2 (y)
    return a * (y**n) / (S**n + y**n) + b * (S**n) / (S**n + x**n) - k * y

def jacobian(x, y, a, b):
    # Derivadas parciales analíticas de las ecuaciones de Hill
    # df/dx
    dfdx = a * (n * x**(n-1) * S**n) / (S**n + x**n)**2 - k
    # df/dy
    dfdy = -b * (n * y**(n-1) * S**n) / (S**n + y**n)**2
    # dg/dx
    dgdx = -b * (n * x**(n-1) * S**n) / (S**n + x**n)**2
    # dg/dy
    dgdy = a * (n * y**(n-1) * S**n) / (S**n + y**n)**2 - k
    
    return np.array([[dfdx, dfdy], [dgdx, dgdy]])

# ============================================================
# 2. INTEGRACIÓN DE UNA TRAYECTORIA (RK4) EN 2D
# ============================================================

def rk4_trajectory(f, g, x0, y0, a, b, t0, t1, dt):
    n_steps = int((t1 - t0) / dt) + 1
    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    t[0], x[0], y[0] = t0, x0, y0

    for i in range(n_steps - 1):
        xi, yi = x[i], y[i]
        
        # Límite de seguridad para evitar overflow en Python
        if abs(xi) > 50 or abs(yi) > 50:
            x[i+1], y[i+1], t[i+1] = xi, yi, t[i] + dt
            continue

        k1x = f(xi, yi, a, b)
        k1y = g(xi, yi, a, b)

        k2x = f(xi + 0.5 * dt * k1x, yi + 0.5 * dt * k1y, a, b)
        k2y = g(xi + 0.5 * dt * k1x, yi + 0.5 * dt * k1y, a, b)

        k3x = f(xi + 0.5 * dt * k2x, yi + 0.5 * dt * k2y, a, b)
        k3y = g(xi + 0.5 * dt * k2x, yi + 0.5 * dt * k2y, a, b)

        k4x = f(xi + dt * k3x, yi + dt * k3y, a, b)
        k4y = g(xi + dt * k3x, yi + dt * k3y, a, b)

        x[i+1] = xi + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        y[i+1] = yi + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
        t[i+1] = t[i] + dt

    return t, x, y

# ============================================================
# 3. BÚSQUEDA NUMÉRICA MÚLTIPLE DE PUNTOS FIJOS
# ============================================================

def find_fixed_points_numeric(a, b):
    # La función objetivo para fsolve
    def system(vars):
        x, y = vars
        return [f(x, y, a, b), g(x, y, a, b)]
    
    # Múltiples estimaciones iniciales para capturar todos los posibles puntos fijos
    guesses = [
        [0.1, 2.5], [2.5, 0.1],   # Estados diferenciados
        [1.5, 1.5], [0.5, 0.5],   # Estado no diferenciado central
        [1.0, 2.0], [2.0, 1.0],   # Posibles puntos de silla
        [0.0, 0.0]
    ]
    
    roots = []
    for guess in guesses:
        root, infodict, ier, mesg = fsolve(system, guess, full_output=True)
        # ier == 1 significa convergencia exitosa. Evitamos concentraciones negativas.
        if ier == 1 and root[0] >= -1e-4 and root[1] >= -1e-4:
            # Forzamos ceros limpios
            root = np.maximum(root, 0.0)
            # Evitar raíces duplicadas
            is_new = True
            for r in roots:
                if np.linalg.norm(root - r) < 1e-3:
                    is_new = False
                    break
            if is_new:
                roots.append(root)
                
    return roots

# ============================================================
# 4. LÓGICA DE DIBUJO DEL PLANO DE FASES INTERACTIVO
# ============================================================

class InteractiveHillPlot:
    def __init__(self, f, g, a_init, b_init, x0_init, y0_init, xlim, ylim):
        self.f = f
        self.g = g
        self.xlim = xlim
        self.ylim = ylim
        
        self.n_field, self.n_nullcline = 30, 400
        self.t0, self.t1, self.dt = 0.0, 100.0, 0.05
        
        self.fig = plt.figure(figsize=(16, 9))
        
        self.ax_plot = self.fig.add_axes([0.05, 0.1, 0.60, 0.8])  
        self.ax_ctrls = self.fig.add_axes([0.68, 0.1, 0.30, 0.8]) 

        self.ax_ctrls.set_axis_off()
        self.rect_ctrls = patches.Rectangle((0, 0), 1, 1, color='#f0f0f0', transform=self.ax_ctrls.transAxes, zorder=0)
        self.ax_ctrls.add_patch(self.rect_ctrls)
        self.ax_ctrls.text(0.5, 0.95, "MODELO DE HILL\n(Paisaje Waddington)", fontsize=16, fontweight='bold', ha='center', va='top', transform=self.ax_ctrls.transAxes)

        self.x_n = np.linspace(xlim[0], xlim[1], self.n_nullcline)
        self.y_n = np.linspace(ylim[0], ylim[1], self.n_nullcline)
        self.Xn, self.Yn = np.meshgrid(self.x_n, self.y_n)

        slider_x = 0.76      
        slider_width = 0.15  
        slider_height = 0.03
        
        self.ax_a = self.fig.add_axes([slider_x, 0.75, slider_width, slider_height])
        self.slider_a = Slider(self.ax_a, 'Autoactivación $a$', 0.0, 2.0, valinit=a_init, color='purple')
        
        self.ax_b = self.fig.add_axes([slider_x, 0.68, slider_width, slider_height])
        self.slider_b = Slider(self.ax_b, 'Inhibición $b$', 0.0, 2.0, valinit=b_init, color='m')
        
        self.ax_x0 = self.fig.add_axes([slider_x, 0.61, slider_width, slider_height])
        self.slider_x0 = Slider(self.ax_x0, 'Inicio Gen 1 $x_0$', xlim[0], xlim[1], valinit=x0_init, color='orange')
        
        self.ax_y0 = self.fig.add_axes([slider_x, 0.54, slider_width, slider_height])
        self.slider_y0 = Slider(self.ax_y0, 'Inicio Gen 2 $y_0$', ylim[0], ylim[1], valinit=y0_init, color='orange')

        self.txt_stability = self.ax_ctrls.text(0.05, 0.45, "", fontsize=10, family='monospace', va='top', transform=self.ax_ctrls.transAxes)

        self.slider_a.on_changed(self.update)
        self.slider_b.on_changed(self.update)
        self.slider_x0.on_changed(self.update)
        self.slider_y0.on_changed(self.update)

        self.update(None)

    def draw_phase_plane(self, a_val, b_val, x0_val, y0_val):
        self.ax_plot.clear()

        # 1. Dibujar el Campo Vectorial
        x_q = np.linspace(self.xlim[0], self.xlim[1], self.n_field)
        y_q = np.linspace(self.ylim[0], self.ylim[1], self.n_field)
        Xq, Yq = np.meshgrid(x_q, y_q)

        U = self.f(Xq, Yq, a_val, b_val)
        V = self.g(Xq, Yq, a_val, b_val)
        M = np.sqrt(U**2 + V**2)
        M_safe = np.where(M == 0, 1.0, M)
        U_plot, V_plot = U / M_safe, V / M_safe

        self.ax_plot.quiver(Xq, Yq, U_plot, V_plot, M, angles='xy', cmap='plasma', alpha=0.5)

        # 2. Dibujar Nulclinas
        Fn = self.f(self.Xn, self.Yn, a_val, b_val)
        Gn = self.g(self.Xn, self.Yn, a_val, b_val)
        self.ax_plot.contour(self.Xn, self.Yn, Fn, levels=[0], colors='purple', linewidths=2.0)
        self.ax_plot.contour(self.Xn, self.Yn, Gn, levels=[0], colors='m', linewidths=2.0)

        # 3. Encontrar y clasificar Puntos Fijos
        fixed_points = find_fixed_points_numeric(a_val, b_val)
        
        stability_str = f"ANÁLISIS DE PUNTOS FIJOS:\n"
        stability_str += f"-------------------------\n"
        
        for i, fp in enumerate(fixed_points):
            fp_x, fp_y = fp
            J = jacobian(fp_x, fp_y, a_val, b_val)
            trace = np.trace(J)
            det = np.linalg.det(J)
            
            # Criterios de estabilidad en 2D (Sistemas continuos)
            if trace < 0 and det > 0:
                status = "ESTABLE (Valle)"
                fp_color = 'blue'
            elif det < 0:
                status = "SILLA (Inestable)"
                fp_color = 'green'
            else:
                status = "INESTABLE (Cima)"
                fp_color = 'red'
                
            self.ax_plot.scatter(fp_x, fp_y, color=fp_color, s=100, marker='o', zorder=5, edgecolor='white', linewidth=1.5)
            stability_str += f"PF {i+1}: ({fp_x:.2f}, {fp_y:.2f}) -> {status}\n"

        self.txt_stability.set_text(stability_str)

        # 4. Integrar e imprimir Trayectoria RK4
        t, xt, yt = rk4_trajectory(self.f, self.g, x0_val, y0_val, a_val, b_val, self.t0, self.t1, self.dt)
        self.ax_plot.plot(xt, yt, color='black', linewidth=2.5, zorder=4)
        self.ax_plot.scatter([x0_val], [y0_val], color='orange', s=120, marker='X', zorder=6, edgecolor='black')

        # 5. Leyendas y Estilos
        legend_elements = [
            Line2D([0], [0], color='purple', lw=2.0, label="Nulclina $dx_1/dt = 0$"),
            Line2D([0], [0], color='m', lw=2.0, label="Nulclina $dx_2/dt = 0$"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Estado Estable (Diferenciado/Madre)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Punto de Silla (Transición)'),
            Line2D([0], [0], color='black', lw=2.5, label='Evolución Temporal (Trayectoria)'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='orange', markeredgecolor='black', markersize=12, label='Célula Inicial')
        ]
        self.ax_plot.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

        self.ax_plot.set_xlim(self.xlim)
        self.ax_plot.set_ylim(self.ylim)
        self.ax_plot.set_xlabel("Expresión Gen 1 ($x_1$)", fontsize=12)
        self.ax_plot.set_ylabel("Expresión Gen 2 ($x_2$)", fontsize=12)
        self.ax_plot.set_title(f"Dinámica de Diferenciación Celular\nParámetros: $a = {a_val:.2f}$, $b = {b_val:.2f}$", fontsize=14, fontweight='bold')
        
        self.ax_plot.axhline(0, color='black', linewidth=1.0)
        self.ax_plot.axvline(0, color='black', linewidth=1.0)
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.set_aspect('equal')

    def update(self, val):
        a_val = self.slider_a.val
        b_val = self.slider_b.val
        x0_val = self.slider_x0.val
        y0_val = self.slider_y0.val
        self.draw_phase_plane(a_val, b_val, x0_val, y0_val)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

# ============================================================
# 5. EJECUCIÓN PRINCIPAL Y ANIMACIÓN
# ============================================================

if __name__ == '__main__':
    print("Inicializando simulador del Paisaje de Waddington...")
    
    a_orig = 1.48
    
    # Arrancamos con un valor de 'a' donde existen 3 puntos estables
    HillApp = InteractiveHillPlot(
        f, g,
        a_init=a_orig,   
        b_init=1.0,    # Fijo según la literatura
        x0_init=1.4, 
        y0_init=1.6,
        xlim=(-0.1, 3.0),
        ylim=(-0.1, 3.0)
    )

    respuesta = input("¿Deseas generar un GIF con la animación variando el parámetro a (bifurcación)? (s/n): ")
    
    if respuesta.lower() == 's':
        print("Preparando la animación (esto tardará unos momentos calculando los estados y trayectorias)...")
        
        # Diseñamos la coreografía de 'a' manteniendo 'b' fijo en 1.0
        # Observaremos cómo al bajar de 0.78, el punto central se vuelve inestable (verde/rojo)
        a_frames = np.concatenate([
            np.linspace(2.0, 0.2, 50),       # 'a' baja lentamente pasando por la bifurcación
            np.linspace(0.2, a_orig, 20),    # 'a' vuelve a su posición original
            np.ones(10) * a_orig             # Pausa final
        ])
        
        def update_anim(frame):
            a_val = a_frames[frame]
            # Mover el slider automáticamente redibuja todo
            HillApp.slider_a.set_val(a_val)
            return []
            
        anim = animation.FuncAnimation(HillApp.fig, update_anim, frames=len(a_frames), interval=100)
        
        # Guardamos el GIF usando Pillow
        anim.save('animacion_waddington_a.gif', writer='pillow')
        print("¡GIF guardado con éxito como 'animacion_waddington_a.gif'!")

    print("Abriendo la ventana interactiva. Prueba a mover el slider 'a'...")
    # Restauramos a su valor original por si la animación lo dejó alterado
    HillApp.slider_a.set_val(a_orig)
    HillApp.show()
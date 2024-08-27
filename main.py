import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
from scipy.linalg import inv

# Parâmetros do sistema quarter-car
m_sprung = 250     # Massa suspensa (corpo do carro) (kg)
m_unsprung = 50    # Massa não suspensa (roda) (kg)
k_spring = 15000   # Constante da mola da suspensão (N/m)
k_tire = 200000    # Constante da mola do pneu (N/m)
b_damper = 1000    # Coeficiente de amortecimento da suspensão (Ns/m)
f_road = 0.1       # Amplitude de distúrbio da estrada (m)

# Matrizes de estado do sistema
A = np.array([[0, 1, 0, 0],
              [-k_spring/m_unsprung, -b_damper/m_unsprung, k_spring/m_unsprung, 0],
              [0, 0, 0, 1],
              [k_spring/m_sprung, 0, -k_spring/m_sprung, -b_damper/m_sprung]])

B = np.array([[0], [1/m_unsprung], [0], [-1/m_sprung]])

C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
D = np.array([[0], [0]])

# Função para o controlador LQR
def lqr(A, B, Q, R):
    # Resolver a equação de Riccati
    P = solve_continuous_are(A, B, Q, R)
    # Calcular os ganhos LQR
    K = inv(R) @ B.T @ P
    return K

# Matrizes Q e R para o LQR
Q = np.diag([100, 1, 100, 1])
R = np.array([[1]])

# Obter os ganhos do controlador LQR
K = lqr(A, B, Q, R)

# Dinâmica do sistema passivo
def passive_system(x, t):
    u = f_road if t > 0 else 0
    dxdt = A @ x + B.flatten() * u
    return dxdt

# Dinâmica do sistema ativo
def active_system(x, t):
    u = f_road if t > 0 else 0
    control_input = -K @ x
    dxdt = (A - B @ K) @ x + B.flatten() * u
    return dxdt

# Condições iniciais
x0 = [0, 0, 0, 0]

# Vetor de tempo
t = np.linspace(0, 5, 500)

# Resolver para o sistema passivo
x_passive = odeint(passive_system, x0, t)

# Resolver para o sistema ativo
x_active = odeint(active_system, x0, t)

# Plotar os resultados
plt.figure(figsize=(10, 8))

# Deslocamento do corpo
plt.subplot(2, 1, 1)
plt.plot(t, x_passive[:, 0], 'b', label='Suspensão Passiva')
plt.plot(t, x_active[:, 0], 'r--', label='Suspensão Ativa')
plt.title('Deslocamento do Corpo do Carro')
plt.xlabel('Tempo (s)')
plt.ylabel('Deslocamento (m)')
plt.legend()

# Deslocamento da roda
plt.subplot(2, 1, 2)
plt.plot(t, x_passive[:, 2], 'b', label='Suspensão Passiva')
plt.plot(t, x_active[:, 2], 'r--', label='Suspensão Ativa')
plt.title('Deslocamento da Roda')
plt.xlabel('Tempo (s)')
plt.ylabel('Deslocamento (m)')
plt.legend()

plt.tight_layout()
plt.show()

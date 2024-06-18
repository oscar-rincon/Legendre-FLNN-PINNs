import torch
import matplotlib.pyplot as plt
import torch
from torch.optim import LBFGS
import time
import os
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definición de la función de Legendre
def legendre(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2.0 * n - 1.0) * x * legendre(n - 1, x) - (n - 1) * legendre(n - 2, x)) / n
    
    
# Evaluación de la serie de Legendre 2-D utilizando broadcasting
def evaluate_legendre_series(coefficients, leg_x, leg_y):
    n = int(torch.sqrt(torch.tensor(coefficients.numel()).float()))  # Convert to tensor before sqrt
    coefficients = coefficients.view(n, n)
    
    # Utilizando broadcasting para calcular la serie de Legendre
    result = torch.sum(coefficients[:, :, None, None] * leg_x[:, None, :, :] * leg_y[None, :, :, :], dim=(0, 1))
    
    return result

# Precomputar leg_x y leg_y una vez
start_precompute = time.time()

# Elige una suposición inicial para los coeficientes
N = 10  # Grado máximo de los polinomios de Legendre
initial_guess = torch.zeros(N*N, requires_grad=True, device=device)  # Suposición inicial para los coeficientes

# Datos de entrada para la función y los polinomios de Legendre
X1, X2 = torch.meshgrid(torch.linspace(0, 1, 100, device=device), torch.linspace(0, 1, 100, device=device))
 
n = int(torch.sqrt(torch.tensor(initial_guess.numel(), device=device).float()))
leg_x = torch.stack([legendre(i, X1).to(device) for i in range(n)], dim=0)
leg_y = torch.stack([legendre(j, X2).to(device) for j in range(n)], dim=0)
end_precompute = time.time()

# Medir el tiempo de precomputación
time_precompute = end_precompute - start_precompute
print(f"Tiempo de precomputación de Legendre polynomials: {time_precompute:.4f} segundos")


# Define la función para la cual quieres calcular los coeficientes de la serie de Legendre
def f(x1, x2):
    #return torch.sin(x1) * torch.cos(x2) + 0.5 * torch.cos(2*x1) * torch.cos(2*x2) + 0.25 * torch.cos(4*x1) * torch.cos(4*x2)
    return 16*(1 - x1) * x1 * (1 - x2) * x2

# Define la función de error
def error_function(coefficients, leg_x, leg_y):
    approximation = evaluate_legendre_series(coefficients, leg_x, leg_y)
    error = torch.sum((f(X1, X2) - approximation) ** 2)
    return error

# Usar una función de optimización para minimizar la función de error con respecto a los coeficientes
optimizer = LBFGS([initial_guess], lr=1)

def closure():
    optimizer.zero_grad()
    loss = error_function(initial_guess, leg_x, leg_y)
    loss.backward()
    return loss

# Realizar pasos de optimización
start_optimization = time.time()
for _ in range(10):  # Número de pasos de optimización
    optimizer.step(closure)
end_optimization = time.time()

# Coeficientes óptimos encontrados
optimal_coefficients = initial_guess.detach()

# Evaluar la aproximación de la serie de Legendre
start_evaluation = time.time()
approximation = evaluate_legendre_series(optimal_coefficients, leg_x, leg_y)
end_evaluation = time.time()

# Medir el tiempo de optimización y evaluación
time_optimization = end_optimization - start_optimization
time_evaluation = end_evaluation - start_evaluation

print(f"Tiempo de optimización: {time_optimization:.4f} segundos")
print(f"Tiempo de evaluación: {time_evaluation:.4f} segundos")

# Graficar el resultado usando contourf
plt.figure(figsize=(8, 6))
plt.contourf(X1.cpu().numpy(), X2.cpu().numpy(), approximation.cpu().numpy(), cmap='viridis')
plt.colorbar(label='Approximation')
plt.title('Approximation of f(x1, x2) using Legendre Series')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()    
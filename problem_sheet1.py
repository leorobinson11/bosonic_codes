from qutip import coherent, fock, destroy, num, qeye, Qobj
from qutip.visualization import plot_fock_distribution

from matplotlib import pyplot as plt
import numpy as np
import math

""" Codewords """

def two_legged_cat(N_max, alpha):
    assert alpha >= 1 and alpha <= 3, "alpha needs to be in interval [1,3]"
    a0 = coherent(N_max, alpha)
    a1 = coherent(N_max, -alpha)
    logical0 = (a0 + a1).unit()
    logical1 = (a0 - a1).unit()
    return logical0, logical1


def four_legged_cat(N_max, alpha):
    assert alpha >= 1 and alpha <= 3, "alpha needs to be in interval [1,3]"
    a0 = coherent(N_max, alpha)
    a1 = coherent(N_max, 1j*alpha)
    a2 = coherent(N_max, -alpha)
    a3 = coherent(N_max, -1j*alpha)
    logical0 = (a0 + a1 + a2 + a3).unit()
    logical1 = (a0 - a1 + a2 - a3).unit()
    return logical0, logical1

def binomial_ND_22(N_max):
    logical0 = np.sqrt(1/2)*(fock(N_max, 0) + fock(N_max, 4))
    logical1 = fock(N_max, 2)
    return logical0, logical1

def binomial_ND_024(N_max):
    logical0 = 0.5*fock(N_max, 0) + np.sqrt(3/4)*fock(N_max, 4)
    logical1 = np.sqrt(3/4)*fock(N_max, 2) + 0.5*fock(N_max, 6)
    return logical0, logical1

def binomial_ND_31(N_max):
    logical0 = np.sqrt(1/2)*(fock(N_max, 0) + fock(N_max,6))
    logical1 = fock(N_max, 3)
    return logical0, logical1


""" Goetsmann Knill Matrix """

def kl_matrix(errors, codewords):
    n = len(errors)
    m = len(codewords)
    kl = np.zeros((n, n, m, m), dtype=complex)
    for a in range(n):
        for b in range(n):
            op = errors[a].dag()*errors[b]
            for i in range(m):
                for j in range(m):
                    kl[a, b, i, j] = (codewords[i].dag()*op*codewords[j]).full()[0, 0]
    return kl

def kl_violation_offdiagonal(kl):
    n, _, d, _ = kl.shape
    total = 0.0
    for a in range(n):
        for b in range(n):
            for mu in range(d):
                for nu in range(d):
                    if mu != nu:
                        total += abs(kl[a, b, mu, nu])**2
    return np.sqrt(total)

def kl_violation_diagonal(kl):
    n, _, d, _ = kl.shape
    total = 0.0
    for a in range(n):
        for b in range(n):
            # average over diagonal code entries
            alpha_ab = np.mean([kl[a, b, mu, mu] for mu in range(d)])
            for mu in range(d):
                total += abs(kl[a, b, mu, mu] - alpha_ab)**2
    return np.sqrt(total)

def plot_kl(kl):
    plt.imshow(np.abs(kl).reshape(kl.shape[0]*kl.shape[1], -1),cmap='gray_r', vmin=0)
    plt.colorbar(label='|KL|')
    plt.show()


""" Photon-Loss Channel """

def loss_kraus_operators(N_max, l_max, eta):
    operators = []
    a = destroy(N_max)
    eta_n_half = Qobj(np.diag(eta ** (np.arange(N_max) / 2)))
    for l in range(l_max+1):
        E = np.sqrt((1-eta)**l/math.factorial(l))*eta_n_half*a**l
        operators.append(E)
    return operators

def recovery_operators(N_max, l_max, logical0, logical1):
    a = destroy(N_max)
    operators = []
    for l in range(l_max):
        w0 = (a**l*logical0).unit()
        w1 = (a**l*logical1).unit()
        R = logical0 * w0.dag() + logical1 * w1.dag()
        operators.append(R)
    return operators

def error_words(N_max, l, logical0, logical1):
    a = destroy(N_max)
    w0 =  a**l*logical0
    w1 = a**l*logical1
    return w0, w1

def avrg_fidelity(A, B):
    r = A/B
    return (r**4 + 4*r**3 - 4*r - 1 + 4*np.log(r)*r**2)/(2*(r-1)*(r+1)**3)

def plot_fidelity_after_loss(N_max, logical0, logical1, code_name=""):
    F0 = [] # fidelities for no photon loss
    F1 = [] # fidelities for single photon loss

    w00, w01 = error_words(N_max, 0, logical0, logical1)
    w10, w11 = error_words(N_max, 1, logical0, logical1)

    kt_range = np.linspace(0, 0.2, 100)
    for kt in kt_range:
        eta = np.exp(-kt)
        E0, E1 = loss_kraus_operators(N_max, 1, eta)
       
        # Fidelities avaraged over blochsphere
        A0 = (w00.dag() * E0 * logical0).full()[0, 0]
        B0 = (w01.dag() * E0 * logical1).full()[0, 0]
        F0.append(avrg_fidelity(A0, B0))

        A1 = (w10.dag() * E1 * logical0).full()[0, 0]
        B1 = (w11.dag() * E1 * logical1).full()[0, 0]
        F1.append(avrg_fidelity(A1, B1))

    plt.plot(kt_range, F0, label='k=0')
    plt.plot(kt_range, F1, label='k=1')
    plt.title(f"Recovery of {code_name} after photon loss channel")
    plt.xlabel("kt")
    plt.ylabel("Avarage Fidelity")
    plt.legend()  
    plt.tight_layout()
    plt.savefig(
        f"Photon loss channel {code_name}.pdf",
        format="pdf",
        bbox_inches="tight"
    )


""" Dephansing """

def dephasing_kraus_operators(N_max, l_max, lam):
    operators = []
    n = num(N_max)
    exp_n_squared = Qobj(np.diag(np.exp(-0.5*lam*np.arange(N_max)**2)))
    for l in range(l_max+1):
        E = np.sqrt((lam)**l/math.factorial(l))*exp_n_squared*n**l
        operators.append(E)
    return operators

def plot_fidelity_after_dephasing(N_max, logical0, logical1, code_name=""):
    F0 = [] 
    F1 = [] 

    kt_range = np.linspace(0, 0.2, 100)
    for kt in kt_range:
        E0, E1 = dephasing_kraus_operators(N_max, 1, kt)
       
        # Fidelities avaraged over blochsphere
        A0 = (logical0.dag() * E0 * logical0).full()[0, 0]
        B0 = (logical1.dag() * E0 * logical1).full()[0, 0]
        F0.append(avrg_fidelity(A0, B0))

        A1 = (logical0.dag() * E1 * logical0).full()[0, 0]
        B1 = (logical1.dag() * E1 * logical1).full()[0, 0]
        F1.append(avrg_fidelity(A1, B1))

    plt.plot(kt_range, F0, label='k=0')
    plt.plot(kt_range, F1, label='k=1')
    plt.title(f"Recovery of {code_name} after dephansing channel")
    plt.xlabel("kt")
    plt.ylabel("Avarage Fidelity")
    plt.legend()  
    plt.tight_layout()
    plt.savefig(
        f"Dephasing channel {code_name}.pdf",
        format="pdf",
        bbox_inches="tight"
    )
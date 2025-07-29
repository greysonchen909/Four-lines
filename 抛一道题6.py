import numpy as np
from scipy.stats import norm

def calculate_partial_factors(mu_D, sigma_D, u, alpha, mu_lnS, sigma_lnS, R_k, beta, max_iter=100, tol=1e-3):
    # 初始化分项系数
    gamma = np.array([1.0, 1.0, 1.0])
    
    # 雪荷载S的当量正态参数（固定）
    mu_S_N = np.exp(mu_lnS + 0.5 * sigma_lnS**2)
    sigma_S_N = mu_S_N * np.sqrt(np.exp(sigma_lnS**2) - 1)
    
    for iteration in range(max_iter):
        # 内部循环求解U和L的当量参数
        U = np.zeros(3)  # 初始假设U为0
        sigma_L_N = 1.0  # 初始假设
        mu_L_N = 0.0
        
        for _ in range(10):  # 内部迭代
            U2 = U[1]
            # 计算L_star
            cdf_U2 = norm.cdf(U2)
            if cdf_U2 <= 0 or cdf_U2 >= 1:
                cdf_U2 = np.clip(cdf_U2, 1e-10, 1 - 1e-10)
            try:
                L_star = u - (1 / alpha) * np.log(-np.log(cdf_U2))
            except:
                L_star = u  # 处理数值问题
            
            # 计算f_L(L_star)
            z = -alpha * (L_star - u)
            f_L = alpha * np.exp(z - np.exp(z))
            
            # 计算sigma_L_N和mu_L_N
            sigma_L_N = norm.pdf(U2) / f_L
            mu_L_N = L_star - sigma_L_N * U2
            
            # 计算梯度向量和方向余弦
            gradient = np.array([-gamma[0] * sigma_D,
                                -gamma[1] * sigma_L_N,
                                -gamma[2] * sigma_S_N])
            norm_grad = np.linalg.norm(gradient)
            if norm_grad == 0:
                alpha_i = np.zeros(3)
            else:
                alpha_i = gradient / norm_grad
            U_new = -alpha_i * beta
            
            # 检查收敛
            if np.linalg.norm(U_new - U) < 1e-6:
                break
            U = U_new.copy()
        
        # 计算设计点
        D_star = mu_D + sigma_D * U[0]
        S_star = mu_S_N + sigma_S_N * U[2]
        
        # 更新分项系数（最小二乘法）
        A = np.array([D_star, L_star, S_star])
        gamma_old = gamma.copy()
        denominator = np.dot(A, A)
        if denominator == 0:
            gamma_new = gamma_old
        else:
            lambda_val = 2 * (np.dot(gamma_old, A) - R_k) / denominator  # Lagrange方法
            gamma_new = gamma_old - (lambda_val * A) / 2
        
        # 检查收敛
        if np.max(np.abs(gamma_new - gamma_old)) < tol:
            return gamma_new, iteration + 1
        
        gamma = gamma_new
    
    return gamma, max_iter

def get_float_input(prompt):
    """从终端获取浮点数输入"""
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("无效输入，请输入有效的数字")

if __name__ == "__main__":
    # 从终端获取所有参数
    print("请输入结构可靠性分析参数：")
    mu_D = get_float_input("永久荷载均值 mu_D: ")
    sigma_D = get_float_input("永久荷载标准差 sigma_D: ")
    u = get_float_input("极值I型位置参数 u: ")
    alpha = get_float_input("极值I型尺度参数 alpha: ")
    mu_lnS = get_float_input("雪荷载对数均值 mu_lnS: ")
    sigma_lnS = get_float_input("雪荷载对数标准差 sigma_lnS: ")
    R_k = get_float_input("抗力标准值 R_k: ")
    beta = get_float_input("目标可靠指标 beta: ")
    
    # 计算分项系数
    gamma, iterations = calculate_partial_factors(mu_D, sigma_D, u, alpha, mu_lnS, sigma_lnS, R_k, beta)
    
    # 输出结果
    print("\n计算结果：")
    print(f"迭代次数: {iterations}")
    print(f"永久荷载分项系数 γ_D: {gamma[0]:.6f}")
    print(f"可变荷载分项系数 γ_L: {gamma[1]:.6f}")
    print(f"雪荷载分项系数 γ_S: {gamma[2]:.6f}")

    # 实例化输入
    # mu_D = 120.0        永久荷载均值
    # sigma_D = 9.6      永久荷载标准差
    # u = 30.0            极值I型位置参数
    # alpha = 0.25         极值I型尺度参数
    # mu_lnS = 1.0        雪荷载对数均值
    # sigma_lnS = 0.35     雪荷载对数标准差
    # R_k = 280.0         抗力标准值
    # beta = 4.2          目标可靠指标
import numpy as np
def MVMD(signal, alpha, tau, K, DC, init, tol):
    try:assert len(signal.shape)==2
    except: AssertionError:print("Input must be a 2D matrix")
    x, y = signal.shape
    if x > y:
        C = y # number of channels
        T = x # lengthoftheSignal
        signal = signal.T.conjugate()
    else:
        C = x # number of channels
        T = y # length of the Signal
    if T%2!=0:
        signal=signal[:,:-1]
    fs = 1 / T
    # ltemp = T // 2
    # Mirroring
    f=np.append(np.flip(signal[:,:T//2],axis=1),signal,axis=-1)#取整问题
    f=np.append(f,np.flip(signal[:,T//2:T],axis=1),axis=-1)# Time Domain  0 to T(of mirrored signal)
    T = f.shape[1]
    t = np.arange(1,T+1)/T
    # frequencies
    freqs = t - 0.5 - 1 / T
    # Construct and center
    f_hat = np.fft.fftshift(np.fft.fft(f),1)
    f_hat_plus = f_hat
    f_hat_plus[:, : T // 2] = 0
    # ------------ Initialization
    # Maximum number of iterations
    N = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones((1, K))
    #Alpha = alpha
    # matrix keeping track of every iterant
    u_hat_plus_00 = np.zeros((len(freqs), C, K),dtype=complex)
    u_hat_plus = np.zeros((len(freqs), C, K),dtype=complex)
    omega_plus = np.zeros((N, K))
    if init ==1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * (i)
    elif init==2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K)))
    else: omega_plus[0,:] = 0
    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0
    # start with empty dual variables
    lambda_hat = np.zeros((len(freqs), C, N),dtype=complex)
    # other inits
    uDiff = tol +np.spacing(1) # update  step
    n = 1 # loop counter
    sum_uk = np.zeros((len(freqs), C)) # accumulator
    # --------------- Algorithm of MVMD
    while (uDiff > tol and n < N): # not converged and below iterations limit
        #  update modes
        for k in range(K):
        # update mode accumulator
            if k >= 1:
                sum_uk = u_hat_plus[:,:, k - 1] + sum_uk - u_hat_plus_00[:,:, k]
            else:
                sum_uk = u_hat_plus_00[:,:, -1] + sum_uk - u_hat_plus_00[:,:, k]
            # update spectrum of mode through Wiener filter of residuals
            for c in range(C):
                u_hat_plus[:, c, k] = (f_hat_plus[c,:].T - sum_uk[:,c] - lambda_hat[:,c,n-1]/2)/(1+Alpha[0,k]*(freqs.T - omega_plus[n-1, k])**2)
        #update first omega if not held at 0
            if DC or (k >= 1):# center frequencies
                numerator =np.matmul(freqs[T // 2 :T],(abs(u_hat_plus[T // 2:T,:, k])**2))
                denominator = np.sum(abs(u_hat_plus[T // 2 + 1:T,:, k])**2,0)
                temp1 = np.sum(numerator,0)
                temp2 = np.sum(denominator,0)
                omega_plus[n , k] = temp1 / temp2
        #Dual ascent
        lambda_hat[:,:, n] = lambda_hat[:,:, n-1] + tau * (np.sum(u_hat_plus, 2) - f_hat_plus.T)
        # loop counter
        n = n + 1
        u_hat_plus_m1 = u_hat_plus_00
        u_hat_plus_00 = np.copy(u_hat_plus)
        # converged yet?
        uDiff = u_hat_plus_00 - u_hat_plus_m1
        uDiff = 1 / T * (uDiff) * np.conj(uDiff)
        uDiff = np.spacing(1) + abs(np.sum(uDiff))
    N = min(N, n)
    omega = omega_plus[:N,:]
    #Signal reconstruction
    u_hat = np.zeros((T, K, C),dtype=complex)
    for c in range(C):
        u_hat[T//2: T,:, c] = u_hat_plus[T//2 : T, c,:]
        u_hat[1:T//2+1,:, c] = np.flip(np.conj(u_hat_plus[T//2 : T, c,:]),axis=0)
        u_hat[0,:, c] = np.conj(u_hat[-1,:, c])
    u = np.zeros((K, len(t), C))
    for k in range(K):
        for c in range(C):
            u[k,:, c]=np.fft.ifft(np.fft.ifftshift(u_hat[:, k, c])).real
    u = u[:, T // 4 : 3 * T // 4,:]
    u_hat = np.zeros([u.shape[1], K,C], dtype=complex)
    for k in range(K):
        for c in range(C):
            u_hat[:, k, c]=np.fft.fftshift(np.fft.fft(u[k,:, c])).T.conjugate()
    u_hat = np.transpose(u_hat, [1,0,2])
    return u,u_hat,omega
# testsignal=[0.27, 0.25, 0.23, 0.24, 0.23, 0.23, 0.34, 0.26, 0.23, 0.33, 0.44, 0.49, 0.7, 1.19, 0.96, 0.77, 0.8, 0.7, 0.84, 1.17, 1.1, 0.91, 0.98, 0.7, 0.39, 0.36, 0.35, 0.27, 0.23, 0.22, 0.23, 0.22, 0.32, 0.29, 1.18, 0.57, 0.44, 0.6, 0.63, 0.35, 0.61, 0.41, 0.81, 1.15, 1.01, 0.83, 0.62, 0.37, 0.44, 0.42, 0.34, 0.25, 0.24, 0.23, 0.24, 0.23, 0.65, 0.21, 0.21, 0.33, 0.24, 0.36, 0.44, 0.9, 0.61, 0.44, 2.02, 1.71, 1.34, 1.3, 1.14, 0.67, 0.49, 0.41, 0.29, 0.28, 0.24, 0.25, 0.23, 0.24, 0.47, 0.22, 0.21, 0.8, 0.88, 0.33, 0.23, 0.21, 0.64, 3.65, 1.24, 1.45, 1.14, 1.01, 0.73, 0.4, 0.41, 0.3, 0.28, 0.24, 0.24, 0.24, 0.24, 0.23, 0.47, 0.65, 0.21, 0.21, 0.2, 0.37, 0.26, 0.87, 0.72, 0.71, 0.88, 1.04, 1.37, 1.6, 0.59, 0.33, 0.33, 0.28, 0.27, 0.23, 0.24, 0.23, 0.24, 0.23, 0.51, 0.33, 0.21, 0.24, 0.21, 0.31, 0.77, 1.0, 0.21, 0.2, 0.21, 0.2, 0.42, 1.44, 0.96, 0.55, 0.42, 0.42, 0.34, 0.24, 0.24, 0.22, 0.23, 0.22, 1.04, 0.24, 0.2, 0.21, 0.2, 0.38, 0.98, 0.8, 0.52, 0.63, 2.29, 3.2, 1.28, 0.71, 0.59, 0.36, 0.36, 0.29, 0.23, 0.24, 0.23, 0.24, 0.23, 0.24, 0.56, 0.24, 0.35, 1.15, 2.98, 2.33, 3.21, 0.97, 0.67, 0.53, 1.14, 1.75, 5.13, 1.07, 1.18, 0.59, 0.37, 0.3, 0.25, 0.24, 0.23, 0.23, 0.25, 0.23, 0.35, 0.43, 1.1, 1.49, 0.65, 3.16, 2.31, 0.88, 0.8, 0.69, 0.24, 0.9, 1.13, 1.79, 1.47, 0.97, 0.32, 0.3, 0.31, 0.24, 0.24, 0.23, 0.24, 0.23, 0.44, 0.25, 0.24, 0.23, 0.28, 0.35, 0.36, 0.65, 0.73, 0.75, 0.63, 0.62, 1.61, 1.63, 0.65]
#
# test1=[0.24, 0.19, 0.22, 0.2 , 0.29, 0.23, 0.22, 0.33, 0.32, 0.44, 0.65,0.86, 1.02, 0.52, 0.43, 0.21, 0.57, 1.13, 1.61, 1.4 , 0.62, 0.5 ,0.68, 0.73, 0.65, 0.44, 0.29, 0.2 , 0.22, 0.23, 0.24, 0.22, 0.37, 0.38, 0.56, 0.7 , 0.25, 0.31, 0.2 , 0.25, 0.42, 1.15, 1.29, 1.3 , 2.77, 2.19, 1.44, 1.1 , 2.18, 1.51, 0.21, 0.18, 0.22, 0.25, 0.2 , 0.37, 0.54, 0.81, 0.6 , 0.14, 0.21, 0.18, 0.17, 0.17, 0.19, 0.19, 0.33, 0.46, 0.7 , 0.53, 1.31, 1.09, 0.85, 0.43, 0.25, 0.2 , 0.18, 0.33, 0.24, 0.31, 0.69, 0.83, 0.32, 0.16, 0.23, 0.14, 0.19, 0.24, 0.17, 0.25, 0.43, 0.29, 1.23, 3.11, 1.36, 1.34, 1.37, 1.14, 0.25, 0.14, 0.23, 0.22, 0.2 , 0.39, 0.28, 1.19, 0.88, 0.38, 0.37, 0.38,  0.4 , 0.35, 0.43, 0.41, 0.64, 0.66, 1.51, 1.31, 2.32, 1.33, 0.93,   1.91, 0.21, 0.17, 0.18, 0.23, 0.19, 0.31, 0.71, 0.54, 0.65, 0.2 ,0.17, 0.17, 0.2 , 0.2 , 0.19, 0.24, 0.28, 0.67, 0.91, 0.86, 1.53,   1.63, 1.29, 0.46, 0.24, 0.18, 0.16, 0.28, 0.18, 0.57, 0.73, 0.87, 0.98, 0.18, 0.2 , 0.14, 0.19, 0.22, 0.15, 0.24, 0.3 , 0.3 , 1.15, 2.47, 1.02, 1.06, 1.14, 0.21, 0.21, 0.24, 0.18, 0.28, 0.23, 0.23, 0.34, 0.32, 0.61, 0.66, 0.42, 0.46, 0.28, 0.6 , 0.36, 0.52, 0.84,  0.45, 0.44, 0.52, 0.44, 0.88, 0.92, 0.67, 0.29, 0.16, 0.21, 0.21, 0.23, 0.26, 0.31, 0.3 , 0.25, 0.32, 0.62, 0.59, 0.27, 0.39, 1.39, 1.17, 1.65, 1.39, 1.05, 1.31, 1.09, 1.05, 0.86, 0.38, 0.2 , 0.18,   0.18, 0.2 , 0.22, 0.29, 0.29, 0.87, 0.62, 0.64, 0.19, 0.18, 0.2 , 0.18, 0.21, 0.2 , 0.29, 0.54, 1.29, 1.25, 1.57, 1.31]
# #testsignal=[0.27, 0.25, 0.23, 0.24, 0.23, 0.23, 0.34, 0.26, 0.23, 0.33, 0.44, 0.49, 0.7, 1.19, 0.96, 0.77, 0.8, 0.7, 0.84, 1.17, 1.1, 0.91, 0.98, 0.7, 0.39, 0.36, 0.35, 0.27, 0.23, 0.22, 0.23, 0.22, 0.32, 0.29, 1.18, 0.57, 0.44, 0.6, 0.63, 0.35, 0.61, 0.41, 0.81, 1.15, 1.01, 0.83, 0.62, 0.37, 0.44, 0.42, 0.34, 0.25, 0.24, 0.23, 0.24, 0.23, 0.65, 0.21, 0.21, 0.33, 0.24, 0.36, 0.44, 0.9, 0.61, 0.44, 2.02, 1.71, 1.34, 1.3, 1.14, 0.67, 0.49, 0.41, 0.29, 0.28, 0.24, 0.25, 0.23, 0.24, 0.47, 0.22, 0.21, 0.8, 0.88, 0.33, 0.23, 0.21, 0.64, 3.65, 1.24, 1.45, 1.14, 1.01, 0.73, 0.4, 0.41, 0.3, 0.28, 0.24, 0.24, 0.24, 0.24, 0.23, 0.47, 0.65, 0.21, 0.21, 0.2, 0.37, 0.26, 0.87, 0.72, 0.71, 0.88, 1.04, 1.37, 1.6, 0.59, 0.33, 0.33, 0.28, 0.27, 0.23, 0.24, 0.23, 0.24, 0.23, 0.51, 0.33, 0.21, 0.24, 0.21, 0.31, 0.77, 1.0, 0.21, 0.2, 0.21, 0.2, 0.42, 1.44, 0.96, 0.55, 0.42, 0.42, 0.34, 0.24, 0.24, 0.22, 0.23, 0.22, 1.04, 0.24, 0.2, 0.21, 0.2, 0.38, 0.98, 0.8, 0.52, 0.63, 2.29, 3.2, 1.28, 0.71, 0.59, 0.36, 0.36, 0.29, 0.23, 0.24, 0.23, 0.24, 0.23, 0.24, 0.56, 0.24, 0.35, 1.15, 2.98, 2.33, 3.21, 0.97, 0.67, 0.53, 1.14, 1.75, 5.13, 1.07, 1.18, 0.59, 0.37, 0.3, 0.25, 0.24, 0.23, 0.23, 0.25, 0.23, 0.35, 0.43, 1.1, 1.49, 0.65, 3.16, 2.31, 0.88, 0.8, 0.69, 0.24, 0.9, 1.13, 1.79, 1.47, 0.97, 0.32, 0.3, 0.31, 0.24, 0.24, 0.23, 0.24, 0.23, 0.44, 0.25, 0.24, 0.23, 0.28, 0.35, 0.36, 0.65, 0.73, 0.75, 0.63, 0.62, 1.61, 1.63, 0.65]
# C=np.vstack((testsignal,test1))
# u,u_hat,omega=MVMD(C,500,0,1,0,1,1e-6)



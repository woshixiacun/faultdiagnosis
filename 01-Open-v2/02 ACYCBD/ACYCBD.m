function [h_final, s, kappa, W, count, err, f_est] = ACYCBD(x, fs, N, param, p, K)
    %
    %---------------
    % Input:
    %---------------
    %
    %   x       :       signal to be analyzed
    %   N       :       filter size
    %   param   :       structure of setting parameters organized as follows:
    %                       param.ER    :   minimal relative error on result (default value = 1e-3)
    %                       param.iter  :   maximum number of iterations (default value  = 50)
    %   p       :       cyclostationarity order to maximize (default = 2)
    %   fs      :       sampling frequency of x
    %   K       :       frequency order in EHPS(default = 10)
    %   flim    :       frequency range in EHPS [Hz](default = 30)
    %
    %---------------
    % Output:
    %---------------
    %
    %   h_final :       optimal inverse FIR filter at convergence
    %   s       :       blindly deconvolved signal
    %   kappa   :       value of criterion at convergence
    %   W       :       weights used in the criterion at convergence
    %   count   :       number of iteration to convergence
    %   err     :       relative error on result as a function of iterations
    %   f_est   :       estimated result as a function of iterations
    %
    %---------------
    % Note:
    %---------------
    %       Original CYCBD is coded by J. Antoni and M. Buzzoni.
    %       This ACYCBD code is based on original CYCBD.
    %       Reference
    %           M. Buzzoni, J. Antoni, G. D'Elia,
    %           Blind deconvolution based on cyclostationarity maximization
    %           and its application to fault identification
    %           Journal of Sound And Vibration, 432 (2018) 569-601.
    %---------------
    % Reference:
    %---------------
    %   Author: ¡¾1¡¿B. Zhang, Y. Miao, J. Lin, Y. Yi
    %               "Adaptive maximum second-order cyclostationarity blind deconvolution
    %               and its application for locomotive bearing fault diagnosis"
    %           Mechanical Systems and Signal Processing, 158 (2021) 107736.
    %          ¡¾2¡¿Y. Miao, B. Zhang, J. Lin et al., 
    %             ¡°A review on the application of blind deconvolution in machinery fault diagnosis¡±
    %               Mechanical Systems and Signal Processing, 163 (2022) 108202.
    %-------------------------------------------------
    %
    % Author: @ Boyao Zhang 
    %
    %-------------------------------------------------
    %
    if nargin < 6
        K = 10;
    end

    if nargin < 5
        p = 2;
    end

    if nargin < 4
        param.RE = 1e-3;
        param.iter = 50;
    end

    if nargin < 3
        N = 40;
    end

    L = length(x);
    x = x - mean(x);

    h = zeros(N, 1);
    h(2) = 1;

    XX = CorrMatrix(x, [], N);

    test = 0;
    count = 1;
    kappa_old = 0;
    err = zeros(param.iter, 1);
    f_est = [];

    while test == 0
        s = filter(h, 1, x);
        W = abs(s(N:L)).^p;

        alpha_est = EHPS(s, fs, K);
        f_est = [f_est, alpha_est(1)];
        alpha = alpha_est(1) * (1:100)';

        W = Periodic(W, alpha, fs);
        W = W / (mean(W).^(p / 2));

        XWX = CorrMatrix(x, W, N);
        [h, kappa] = eigs(XWX, XX, 1);
        err(count) = abs(kappa - kappa_old) / abs(kappa_old);

        if (err(count) < param.RE) || count >= param.iter
            test = 1;
        end

        count = count + 1;
        kappa_old = kappa;
    end

    h_final = h;
    count = count - 1;
    s = filter(h, 1, x);
    s = s(N:end);
end

% End of the main function: ACYCBD

%--------------------------------------------------------------------------

function f_est = EHPS(x, fs, K, flim)
    % estimate the period or frequency hidden in x.
    % The sub-function is based on harmonic-related spectrum structure.
    %---------------
    % Input:
    %---------------
    %
    %   x       :       signal to be analyzed
    %   fs      :       sampling frequency of x
    %   K       :       frequency order in EHPS
    %   flim    :       frequency range in EHPS [Hz]
    %
    %---------------
    % Output:
    %---------------
    %
    %   f_est   :       estimated result
    %
    %-------------------------------------------------
    %
    % Author: @ Zhang Boyao
    %
    %-------------------------------------------------
    %
    if nargin < 4
        flim = 300;
    end

    if nargin < 3
        K = 10;
    end

    L = length(x);
    x = x(:);
    x = x - mean(x);

    ex = abs(hilbert(x));
    ex = ex - mean(ex);
    sex = abs(fft(ex, L)) * 2 / L;
    sex = sex(1:round(L / 2));

    ehpsx = ones(round(flim * fs / L), 1);
    ff = sex(2:end);

    for f = 1:length(ehpsx)

        for k = 1:K

            if k * f < fs / 2
                ehpsx(f) = ehpsx(f) * ff(k * f);
            end

        end

    end

    [~, index_max] = max(ehpsx);
    f_est = index_max * (fs / L);

end

% End of the first sub-function: EHPS

%--------------------------------------------------------------------------
function R = CorrMatrix(x, W, N)
    % Compute correlation matrix (NxN) of signal x with weights W
    %---------------
    % Input:
    %---------------
    %
    %   x       :       signal to be analyzed
    %   W       :       the weight vector
    %   N       :       The size of correlation matrix (NxN).
    %
    %---------------
    % Output:
    %---------------
    %
    %   R      :        correlation matrix R_(xx) (NxN) while W = []
    %                   or weighted correlation matrix R_(xwx) (NxN) while W ~= [].
    %-------------------------------------------------
    % Code by J. Antoni and M. Buzzoni
    % Dicember 2017
    %-------------------------------------------------
    L = length(x);
    R = zeros(N);

    if isempty(W)
        W = ones(L - N + 1, 1);
    end

    W = W(:);
    x = x(:);

    for i = 1:N
        R(i, i) = mean(abs(x(N + 1 - i:L + 1 - i)).^2 .* W);

        for j = i + 1:N
            R(i, j) = mean(x(N + 1 - i:L + 1 - i) .* conj(x(N + 1 - j:L + 1 - j)) .* W);
            R(j, i) = conj(R(i, j));
        end

    end

end

% End of the second sub-function: CorrMatrix

%--------------------------------------------------------------------------
function p = Periodic(x, alpha, fs)
    % Extract cyclic components with frequencies alpha in x
    %---------------
    % Input:
    %---------------
    %
    %   x       :       signal to be analyzed
    %   alpha   :       cyclic frequecny vector
    %   fs      :       sampling frequency of x
    %
    %---------------
    % Output:
    %---------------
    %
    %   p       :       cyclic components with frequencies alpha in x
    %
    %-------------------------------------------------
    % Code by J. Antoni and M. Buzzoni
    % Dicember 2017
    %-------------------------------------------------
    x = x(:);
    L = length(x);
    dt = 1 / fs;
    T = dt * L;
    t = (0:dt:T - dt)';
    K = length(alpha);

    alpha(alpha == 0) = [];
    p = mean(x);

    for k = 1:K
        c = mean(x .* exp(-2i * pi * alpha(k) .* t));
        p = p + 2 * real(c * exp(2i * pi * alpha(k) .* t));
    end

    % thresholding for improve weighting matrix
    th = mean(p) + 2 * std(p);
    % th = quantile(p,3)
    % th = th(3)
    p(p < th) = 0;

end

% End of the 3rd sub-function: Periodic

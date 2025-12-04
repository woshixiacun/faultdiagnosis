function [y_final, f_final, ckIter] = imckd(fs, x, filterSize, termIter, T, M)
    %
    %---------------
    % Input:
    %---------------
    %
    %   fs              :       sampling frequency of x
    %   x               :       signal to be analyzed
    %   filterSize      :       filter size
    %   termIter        :       the termination number of iterations
    %   T               :       the signal period
    %   M               :       the shift order
    %
    %---------------
    % Output:
    %---------------
    %
    %   y_final         :       filtered signal
    %   f_final         :       optimal inverse FIR filter at convergence
    %   ckIter          :       Correlated Kurtosis of shift M in iteration
    %
    %---------------
    % Note:
    %---------------
    %       Original mckd is coded by McDonald.
    %       This imckd code is based on original mckd.
    %       Reference
    %           G.L. McDonald, Q. Zhao, M.J. Zuo,
    %               "Maximum correlated Kurtosis deconvolution
    %               and application on gear tooth chip fault detection"
    %           Mechanical Systems and Signal Processing, 33 (2012) 237-255.
    %---------------
    % Reference:
    %---------------
    %       Author: ¡¾1¡¿Y. Miao, M. Zhao, J. Lin, Y. Lei
    %               "Application of an improved
    %               maximum correlated kurtosis deconvolution method
    %               for fault diagnosis of rolling element bearings"
    %           Mechanical Systems and Signal Processing, 92 (2017) 173-195.
    %                 ¡¾2¡¿Y. Miao, M. Zhao, K. Liang, J. Lin
    %             ¡°Application of an improved MCKDA for fault detection 
    %               of wind turbine gear based on encoder signal¡±
    %                 Renewable Energy, 151 (2020) 192-203.
    %-------------------------------------------------
    %
    % Author: @ Yonghao Miao
    %
    %-------------------------------------------------

    % Assign default values for inputs
    if isempty(filterSize)
        filterSize = 100;
    end

    if isempty(termIter)
        termIter = 30;
    end

    if isempty(M)
        M = 3;
    end

    %----------------------------
    if isempty(T)
        xxenvelope = abs(hilbert(x)) - mean(abs(hilbert(x)));
        T = TT(xxenvelope, fs);
    end

    T = round(T);
    TF(1) = T;

    %----------------------------
    x = x(:);
    L = filterSize;
    N = length(x);

    % Calculate XmT
    XmT = zeros(L, N, M + 1);

    for m = 0:M

        for l = 1:L

            if l == 1
                XmT(l, (m * T + 1):end, m + 1) = x(1:N - m * T);
            else
                XmT(l, 2:end, m + 1) = XmT(l - 1, 1:end - 1, m + 1);
            end

        end

    end

    % Calculate the matrix inverse section(X0 and X0^(-1))
    Xinv = inv(XmT(:, :, 1) * XmT(:, :, 1)');

    % Assume initial filter as a delayed impulse
    f = zeros(L, 1);
    f(2) = 1;
    f_best = f;
    ck_best = 0;
    iter_best = 0;

    % Initialize iteration matrices
    y = zeros(N, 1);
    b = zeros(L, 1);
    ckIter = [];
    kurt = [];
    kurt(1) = kurtosis(x);

    % Iteratively adjust the filter to maximize correlated Kurtosis
    n = 1;
    delta = 0;

    while n == 1 || (n <= termIter)

        % Compute output signal
        y = (f' * XmT(:, :, 1))';

        kurt(n + 1) = kurtosis(y);

        % Generate yt
        yt = zeros(N, M);

        for m = 0:M

            if m == 0
                yt(:, m + 1) = y;
            else
                yt(T + 1:end, m + 1) = yt(1:end - T, m);
            end

        end

        % Calculate alpha
        alpha = zeros(N, M + 1);

        for m = 0:M
            alpha(:, m + 1) = (prod(yt(:, [1:m (m + 2):size(yt, 2)]), 2).^2) .* yt(:, m + 1);
        end

        % Calculate beta
        beta = prod(yt, 2);

        % Calculate the sum Xalpha term
        Xalpha = zeros(L, 1);

        for m = 0:M
            Xalpha = Xalpha + XmT(:, :, m + 1) * alpha(:, m + 1);
        end

        % Calculate the new filter coefficients
        f = sum(y.^2) / (2 * sum(beta.^2)) * Xinv * Xalpha;

        f = f / sqrt(sum(f.^2));

        % Calculate the ck value of this iteration
        ckIter(n) = sum(prod(yt, 2).^2) / (sum(y.^2)^(M + 1));

        % Update the best match filter if required
        if ckIter(n) > ck_best
            ck_best = ckIter(n);
            f_best = f;
            iter_best = n;
        end

        %% Update the period
        %%----------------------------
        xyenvelope = abs(hilbert(y)) - mean(abs(hilbert(y)));
        T = TT(xyenvelope, fs);
        T = round(T);
        TF(n + 1) = T;

        % Calculate XmT
        XmT = zeros(L, N, M + 1);

        for m = 0:M

            for l = 1:L

                if l == 1
                    XmT(l, (m * T + 1):end, m + 1) = x(1:N - m * T);
                else
                    XmT(l, 2:end, m + 1) = XmT(l - 1, 1:end - 1, m + 1);
                end

            end

        end

        % Calculate the matrix inverse section
        Xinv = inv(XmT(:, :, 1) * XmT(:, :, 1)');

        %% ----------------------------
        n = n + 1;

        if n == 2
            kurmax = kurt(1);
        elseif kurt(n - 1) > kurmax
            kurmax = kurt(n - 1);
            y_final = y;
            f_best = f;
            f_final = f_best;
            iter_best = n - 1;
        end

    end

    figure;
    [AX, H1, H2] = plotyy(0:n - 1, TF, 0:n - 1, kurt, 'plot');
    set(gcf, 'position', [100, 100, 1000, 600])
    set(AX(1), 'XColor', 'k', 'YColor', 'b');
    set(AX(2), 'XColor', 'k', 'YColor', 'r');

    HH1 = get(AX(1), 'Ylabel');
    set(HH1, 'String', 'Sampling Point');
    set(HH1, 'color', 'b');
    set(HH1, 'FontName', 'Times New Roman');
    set(HH1, 'FontSize', 18);
    set(AX(1), 'ylim', [0 6000]);

    HH2 = get(AX(2), 'Ylabel');
    set(HH2, 'String', 'Kurtosis');
    set(HH2, 'color', 'r');
    set(HH2, 'FontName', 'Times New Roman');
    set(HH2, 'FontSize', 18);
    set(AX(2), 'ylim', [2.9 4.2]);
    set(AX(2), 'FontSize', 18);

    HH3 = get(AX(1), 'Xlabel');
    set(HH3, 'String', 'Iteration');
    set(HH3, 'color', 'k');
    set(HH3, 'FontSize', 18);
    set(HH3, 'FontName', 'Times New Roman');

    set(H1, 'LineStyle', '-');
    set(H1, 'Marker', '*');
    set(H1, 'color', 'b');
    set(H2, 'LineStyle', '-');
    set(H2, 'Marker', 'p');
    set(H2, 'color', 'r');
    setfontsize(18);
    tf = ceil(max(TF) / (20000/29));
    hold on

    for k = 1:tf
        plot(0:n - 1, k * 20000/29 * ones(1, n), 'k:')
    end

    hold off

    disp(['Total iteration number:', num2str(iter_best)]);

end

function [T] = TT (y, fs)
    % estimate the period in y based on auto-correlation function.
    %---------------
    % Input:
    %---------------
    %
    %   y       :       signal to be analyzed
    %   fs      :       sampling frequency
    %
    %---------------
    % Output:
    %---------------
    %
    %   T       :       estimated period in sample
    %
    %-------------------------------------------------
    %
    % Code by Yonghao Miao
    %
    %-------------------------------------------------

    % find the maximum lag M
    M = fs;

    NA = xcorr(y, y, M, 'coeff');
    NA = NA(ceil(length(NA) / 2):end);

    % find first zero-crossing
    sample1 = NA(1);

    for lag = 2:length(NA)
        sample2 = NA(lag);

        if ((sample1 > 0) && (sample2 < 0))
            zeroposi = lag;
            break;
        elseif ((sample1 == 0) || (sample2 == 0))
            zeroposi = lag;
            break;
        else
            sample1 = sample2;
        end

    end

    % Cut from the first zero-crossing
    NA = NA(zeroposi:end);
    % Find the max position (time lag)
    % corresponding the max aside from the first zero-crossing
    [~, max_position] = max(NA);
    % Give the extimated period by autocorrelation
    T = zeroposi + max_position;

end

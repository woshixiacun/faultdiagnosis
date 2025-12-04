function [y_final, f_final, kurtIter] = smhd(fs, x, filterSize, termIter, mu, T, plotMode)
    % sparse maximum harmonics-noise-ratio deconvolution (smhd)
    %
    %%%%%%%%% input %%%%%%%%%
    %   fs:             sampling frequency
    %   x:              input signal,a vector
    %   filterSize:     length of filter
    %   termIter:       maximum number of iterations (default value  = 30)
    %   mu:             initial sparse threshold
    %   T:              prior fault signal period
    %   plotMode:       whether to display the filtered signal (value = 1: Yes/value = 0: No)
    %%%%%%%%% output %%%%%%%%%
    %   y_final:        filtered signal
    %   f_final:        FIR filter at convergence
    %   kurtIter:       number of iteration to convergence
    %
    %----------------------------------
    %
    %   Authors:      Yonghao Miao
    %
    %----------------------------------
    %
    % Reference:
    %%%%%%%%%%%%%%%%
    %
    %          ¡¾1¡¿Y. Miao, M. Zhao, J. Lin, Y. Lei
    %           "Sparse maximum harmonics-to-noise-ratio deconvolution
    %           for weak fault signature detection in bearings".
    %       Measurement Science and Technology, 2016, 27(10)
    %          ¡¾2¡¿Y. Miao, B. Zhang, J. Lin et al., 
    %             ¡°A review on the application of blind deconvolution in machinery fault diagnosis¡±
    %               Mechanical Systems and Signal Processing, 163 (2022) 108202.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%% Initialize parameters %%%%%%%%%
    if isempty(filterSize)
        filterSize = 100;
    end

    if isempty(termIter)
        termIter = 30;
    end

    if isempty(mu)
        mu = mean(x);
    end

    %% SMHD
    x = x(:);
    N = length(x);
    L = filterSize;

    %%%%%%%%% Initialize the fault signal period %%%%%%%%%
    if isempty(T)
        xxenvelope = abs(hilbert(x)) - mean(abs(hilbert(x)));
        [T, ~] = TT(xxenvelope, fs);
    end

    T = round(T);

    autoCorr = zeros(1, L);

    for k = 0:L - 1
        x2 = zeros(N, 1);
        x2(k + 1:end) = x(1:end - k);
        autoCorr(k + 1) = autoCorr(k + 1) + sum(x .* x2);
    end

    A = toeplitz(autoCorr);
    A_inv = inv(2 .* A);

    f = zeros(L, 1);
    y1 = zeros(size(x));
    kurtIter = [];
    hnr = [];
    deltah = [];
    deltak = [];

    % Initialize the filter coefficients
    f(round(L / 2)) = 1;
    f(round(L / 2) + 1) = -1;

    n = 1;
    %%%%%%%%% Iterate to solve the optimal filter coefficients %%%%%%%%%
    while n == 1 || (n <= termIter)
        y = filter(f, 1, x);
        kurtIter(n) = kurtosis(y);
        yenvelope = abs(hilbert(y)) - mean(abs(hilbert(y)));
        [~, hnr(n)] = TT(yenvelope, fs);
        y = y .* (1 - exp(-y.^2 / (2 * mu^2)));
        weightedCrossCorr = zeros(L, 1);

        for k = 0:L - 1
            x2 = zeros(N, 1);
            x3 = zeros(N, 1);
            x2(k + 1:end - T) = x(T + 1:end - k);
            x3(k + 1:end) = x(1:end - k);
            y1(1:end - T) = y(T + 1:end);
            weightedCrossCorr(k + 1) = weightedCrossCorr(k + 1) + ((sum(y .* x2) + sum(y1 .* x3)) .* sum(y.^2)) ./ sum(y .* y1);
        end

        f = A_inv * weightedCrossCorr;
        f = f / sqrt(sum(f.^2));

        n = n + 1;
        %%%%%%%%% Update the sparse threshold  %%%%%%%%%
        [~, temp_hnr] = TT(y, fs);
        deltah(n) = (temp_hnr - hnr(n - 1));
        deltak(n) = (kurtosis(filter(f, 1, x)) / kurtIter(n - 1));

        if deltak(n) > 1
            deltak(n) = 1 + 0.02 * (deltak(n) + 1) / deltak(n);
        else
            deltak(n) = 1 - 0.02 * (deltak(n) + 1) / deltak(n);
        end

        mu = mu * deltak(n);
        % update
        xyenvelope = abs(hilbert(y)) - mean(abs(hilbert(y)));
        [T, ~] = TT(xyenvelope, fs);

        %%%%%%%%%  Determine the maximum number of iterations  %%%%%%%%%
        if n == 2
            hnrmax = hnr(1);
        elseif hnr(n - 1) > hnrmax
            hnrmax = hnr(n - 1);
            y_final = y;
            f_final = f;
        end

    end

    disp(['The number of iteration is ', num2str(n - 1)])
    %%%%%%%%% Display the processed signal %%%%%%%%%
    if plotMode == 1
        figure
        plot((0:length(y_final) - 1) / fs, y_final, 'r')
        title('Filtered signal by SHMD')
        xlabel('Times[s]')
        ylabel('Amplitude[g]')
        set(gcf, 'position', [400, 400, 800, 400])
        legend(['Kurtosis=', num2str(kurtosis(y_final))])

        if n - 1 == termIter
            disp('Terminated for iteration condition.')
        else
            disp('Terminated for minimum change in kurtosis condition.')
        end

    end

end

function [T, HNR] = TT (y, fs)
    % estimate the period in y based on auto-correlation function.
    % calculate the harmonics-to-noise-ratio
    %---------------
    % Input:
    %---------------
    %
    %   y       :       signal to be analyzed
    %   fs      :       sampling frequency of x
    %
    %---------------
    % Output:
    %---------------
    %
    %   T       :       estimated period in sample
    %   HNR     :       harmonics-to-noise ratio
    %
    %-------------------------------------------------
    %
    % Code by Yonghao Miao
    %
    %-------------------------------------------------

    %find the maximum lag M
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
    [max_value, max_position] = max(NA);
    % Give the extimated period by autocorrelation
    T = zeroposi + max_position;
    % Calculate the harmonic energy
    HR = max_value;
    % Calculate the harmonic-to-noise ratio
    HNR = (HR / (1 - HR));

end

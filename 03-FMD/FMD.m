function Final_Mode = FMD(fs, x, FilterSize, CutNum, ModeNum, MaxIterNum)
    %
    %---------------
    % Input:
    %---------------
    %
    %   fs              :       sampling frequency of x
    %   x               :       signal to be analyzed
    %   FilterSiz       :       filter size
    %   CutNum          :       the cut number of the whole frequency band
    %   ModeNum         :       the final mode number
    %   MaxIterNum      :       max iteration number
    %
    %---------------
    % Output:
    %---------------
    %
    %   Final_Mode      :       final mode
    %
    %---------------
    % Reference:
    %---------------
    %       Y. Miao, B. Zhang, C. Li, J. Lin, D. Zhang
    %       "Feature Mode Decomposition:New Decomposition Theory
    %       for Rotating Machinery Fault Diagnosis"
    %       IEEE Transactions on Industrial Electronics.2022
    %       DOI:10.1109/TIE.2022.3156156
    %-------------------------------------------------
    %
    % Author:  Boyao Zhang AND Yonghao Miao
    % Version: 2022/02
    %
    %-------------------------------------------------

    % Initialization  初始化filter
    freq_bound = 0:1 / CutNum:1 - 1 / CutNum;   % 生成一个从 0 开始，均匀分布在[0, 1) 范围内、包含 CutNum个点的归一化频率边界

    temp_filters = zeros(FilterSize, CutNum);
    for n = 1:length(freq_bound)   % 循环会遍历 freq_bound 中的每一个起始频率，并为每个频率段生成一个单独的滤波器
        w = window(@hanning,FilterSize);   %窗函数
        % 生成具有截止频率区间[fl,fu（起始频率 + 步长）的CutNum个 FIR 滤波器
        temp_filters(:, n) = fir1(FilterSize - 1, [freq_bound(n) + eps, freq_bound(n) + 1 / CutNum - eps], w)'; 
    end

    % result initialization
    result = cell(CutNum + 1, 5);
    result(1, :) = {'IterCount', 'Iterations', 'CorrMatrix', 'ComparedModeNum', 'StopNum'};

    % Iterations 更新滤波器
    temp_sig = repmat(x, [1, CutNum]);

    itercount = 2; 
    while 1
        iternum = 2;  %两次迭代后即可快速锁定故障周期
        if itercount == 2
            iternum = MaxIterNum - (CutNum - ModeNum) * iternum;
        end

        result{itercount, 1} = iternum;

        for n = 1:size(temp_filters, 2)  % [1,7]

            f_init = temp_filters(:, n); % 取一个filter
            [y_Iter, f_Iter, k_Iter, T_Iter] = xxc_mckd(fs, temp_sig(:, n), f_init, iternum, [], 1, 0);  %% 滤波器更新与周期估计  
            result{itercount, 2}{n, 1} = y_Iter(:, end);
            result{itercount, 2}{n, 2} = f_Iter(:, end);
            result{itercount, 2}{n, 3} = k_Iter(:, end);
            result{itercount, 2}{n, 4} = abs(fft(f_Iter));
            result{itercount, 2}{n, 4} = result{itercount, 2}{n, 4}(1:FilterSize / 2, :);
            [~, index1] = max(result{itercount, 2}{n, 4});
            result{itercount, 2}{n, 5} = (index1 - 1) * (fs / FilterSize);
            result{itercount, 2}{n, 6} = T_Iter;

        end

        X = result{itercount, 2};

        for n = 1:size(X, 1)
            temp_sig(:, n) = X{n, 1}(:, end);
            temp_filters(:, n) = X{n, 2}(:, end);
        end

        CorrMatrix = abs(corrcoef(temp_sig)); %corrcoef(temp_sig) 会计算 所有信号两两之间的皮尔逊相关系数，得到 N×N 矩阵
        CorrMatrix = triu(CorrMatrix, 1); % 取矩阵 上三角部分（因为是个对称矩阵）

        [I, J, ~] = max_IJ(CorrMatrix);
        Location = [I, J];

        XI = X{I, 1}(:, end);
        XJ = X{J, 1}(:, end);
        XI = XI - mean(XI);
        XJ = XJ - mean(XJ);

        T_1 = result{itercount, 2}{I, 6};
        KI = CK(XI, T_1, 1);
        T_2 = result{itercount, 2}{J, 6};
        KJ = CK(XJ, T_2, 1);

        if KI > KJ
            output = J;
        else
            output = I;
        end

        X(output, :) = [];
        temp_sig(:, output) = [];
        temp_filters(:, output) = [];

        result{itercount, 3} = CorrMatrix;
        result{itercount, 4} = Location;
        result{itercount, 5} = output;

        if size(temp_filters, 2) == ModeNum-1
            break
        end

        itercount = itercount + 1;

    end
    %
    for nn = 1:size(result{itercount, 2}, 1)
        Final_Mode(:, nn) = result{itercount, 2}{nn, 1}(:, end);
    end
end
function [y_final f_final, ckIter T_final] = xxc_mckd(fs, x, f_init, termIter, T, M, plotMode)
    %---------------
    % Reference:
    %---------------
    %   Y. Miao, M. Zhao, J. Lin, Y. Lei
    %       "Application of an improved maximum correlated kurtosis deconvolution method
    %       for fault diagnosis of rolling element bearings"
    %       Mechanical Systems and Signal Processing, 92 (2017) 173 - 195.
    %   Y. Miao, M. Zhao, K. Liang, J. Lin
    %       "Application of an improved MCKDA for fault detection
    %       of wind turbine gear based on encoder signal"
    %       Renewable Energy, 151 (2020) 192 - 203.
    %-------------------------------------------------
    %
    % Author: @ Yonghao Miao
    %
    %-------------------------------------------------

    % Assign default values for inputs

    if (isempty(termIter))
        termIter = 30;
    end
    if (isempty(plotMode))
        plotMode = 0;
    end
    if (isempty(M))
        M = 3;   % M代表移位阶数，决定了在计算 CK 时，需要多少个连续的周期性冲击的乘积来判断信号的周期性
    end

    %%----------------------------
    if (isempty(T))
        xxenvelope = abs(hilbert(x)) - mean(abs(hilbert(x)));  % 计算信号的包络（使用希尔伯特变换的模值），并去均值。
        T = TT(xxenvelope, fs);  % 调用 TT 函数，根据包络的自相关函数估计冲击周期 T（以样本点为单位）。
    end

    T = round(T);

    %%----------------------------
    x = x(:);
    L = length(f_init);  %30
    N = length(x);  %2001

    % Calculate XmT   三维多延时矩阵
    XmT = zeros(L, N, M + 1);  %30x20001x2。用于存储原始信号 x 的多延时矩阵，这是 MCKD 算法的核心。
    for (m = 0:M)  % 延时阶数，从0到M变化。循环生成不同延时m·T下的矩阵。
        for (l = 1:L)  % 滤波器的抽头索引（从 1 到 L 变化）
            if (l == 1)                                         
                XmT(l, (m * T + 1):end, m + 1) = x(1:N - m * T);  % 原始信号 x 沿着时间维度进行周期性延时m·T，并将其填充到多延时矩阵的特定层。
                % 由于信号 x被延时了m·T，所以前 m·T个输出时间点上，数据是缺失的（或应为 0）。因此，数据从第m·T + 1列开始填充，一直到矩阵的最后一列 (end)。
                % 由于目标列只有N - m ·T个位置，所以只取信号 x 的前N - m ·T个样本。
            else                                                  
                XmT(l, 2:end, m + 1) = XmT(l - 1, 1:end - 1, m + 1);  %当前行的元素是前一行的元素向右平移一个位置。
            end
        end
    end

    % Calculate the matrix inverse section
    Xinv = inv(XmT(:, :, 1) * XmT(:, :, 1)');  % (X0X0')的逆，X0是零延时下的X矩阵。

    f = f_init;
    ck_best = 0;

    % Initialize iteration matrices
    y = zeros(N, 1);
    b = zeros(L, 1);
    ckIter = [];

    n = 1;
    %  开始优化
    while n == 1 || (n <= termIter)
        % Compute output signal
        y = (f' * XmT(:, :, 1))'; % 计算零延时下的滤波输出 y

        % Generate yt
        f_final(:, n) = f;
        yt = zeros(N, M); %生成多延时输出矩阵 yt，【每一列】是y信号延时m·T后的版本。【共M+1个延时），yt 实际上应该有M+1列】
        for (m = 0:M)   % 生成M+1个周期延时信号
            if (m == 0)  %零延时
                yt(:, m + 1) = y;
            else   %周期延时
                yt(T + 1:end, m + 1) = yt(1:end - T, m);   % 增加一维，延时T
            end
        end

        % Calculate alpha
        alpha = zeros(N, M + 1);   %计算 MCKD 梯度公式中的alpha项
        for (m = 0:M)
            alpha(:, m + 1) = (prod(yt(:, [1:m (m + 2):size(yt, 2)]), 2).^2) .* yt(:, m + 1);  % uk(n - mTs)
        end                               % 除第 m+1 列以外,所有 yt列的行乘积的平方.再乘以 yt矩阵的第m+1列

        % Calculate beta
        beta = prod(yt, 2);  %计算每行的乘积

        % Calculate the sum Xalpha term
        Xalpha = zeros(L, 1);  % 计算梯度公式中涉及输入信号 X 和 alpha 项的复杂累加项 
        for (m = 0:M)
            Xalpha = Xalpha + XmT(:, :, m + 1) * alpha(:, m + 1);
        end

        % Calculate the new filter coefficients   更新filter
        f = sum(y.^2) / (2 * sum(beta.^2)) * Xinv * Xalpha;   %根据 MCKD 的迭代公式，更新滤波器系数向量 f。
        f = f / sqrt(sum(f.^2));  %L2 范数归一化
        
        % Calculate the ck value of this iteration  计算 峭度 (Kurtosis) 或 峭度变体Ck（MCKD 的目标函数）。
        ckIter(n) = sum(prod(yt, 2).^2) / (sum(y.^2)^(M + 1));  %论文的公式。
        % Update the best match filter if required
        if (ckIter(n) > ck_best)
            ck_best = ckIter(n);  % 如果当前迭代的Ck值 ckIter(n) 优于历史最佳值 ck_best，则更新最佳值
        end

        %% Update the period
        %%----------------------------
        xyenvelope = abs(hilbert(y)) - mean(abs(hilbert(y)));
        T = TT(xyenvelope, fs);
        T = round(T);
        T_final = T;  %更新周期

        % Calculate XmT  更新多时延矩阵（因为周期更新）
        XmT = zeros(L, N, M + 1);
        for (m = 0:M)
            for (l = 1:L)
                if (l == 1)
                    XmT(l, (m * T + 1):end, m + 1) = x(1:N - m * T);
                else
                    XmT(l, 2:end, m + 1) = XmT(l - 1, 1:end - 1, m + 1);
                end
            end
        end

        % Calculate the matrix inverse section
        Xinv = inv(XmT(:, :, 1) * XmT(:, :, 1)');

        %%----------------------------
        y_final(:, n) = filter(f_final(:, n), 1, x);  % 它根据给定的系数 B 和 A 对输入信号 X 进行滤波。
        n = n + 1;
    end
end
function [T] = TT (y, fs)
    % estimate the period in y based on auto-correlation function. 对应论文中“周期估计”相关内容
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

    % find the maximum lag M
    M = fs;  % s这限定了搜索周期T的范围
                                   % 自相关函数 衡量一个信号与其自身在时间延时 t 下的相似程度。
    NA = xcorr(y, y, M, 'coeff');  % 当输入两个相同的信号 y, y 时，它计算的是信号 y 的自相关函数，最大滞后量为 M=fs(即最大滞后量是一个采样周期内的样本数)，并进行归一化。
    NA = NA(ceil(length(NA) / 2):end);  % xcorr这个函数返回的结果包含负滞后、零滞后和正滞后三部分：,截取自相关函数向量 NA 的非负滞后部分
                                        % 由于 xcorr 返回的向量长度是 2M + 1 (从 -M 到 M)，所以长度的一半大致对应于零滞后R(0) 的位置。
    % find first zero-crossing 
    % 在周期性信号分析中，自相关函数的第一个过零点是估计信号周期的一个关键参考点
    sample1 = NA(1);  % 表示自相关函数在零延迟处必然最大且为正，因此从第二个点开始扫描。

    for lag = 2:length(NA)
        sample2 = NA(lag);

        if ((sample1 > 0) && (sample2 < 0)) %第一次由正变负 表明越过了零点，记录该 lag
            zeroposi = lag;
            break;
        elseif ((sample1 == 0) || (sample2 == 0))
            zeroposi = lag;   % 进入周期性振荡区的起点
            break;
        else
            sample1 = sample2;
        end

    end

    % Cut from the first zero-crossing 从第一个过零点开始截取自相关函数。
    NA = NA(zeroposi:end);   
    % Find the max position (time lag)
    % corresponding the max aside from the first zero-crossing  找到截取后的自相关函数中的最大值（第一个过零点之后的第一个峰值）。
    [~, max_position] = max(NA);  %在该区域寻找第一个主峰（通常对应一个周期）, max_position(相对zeroposi)
    % Give the extimated period by autocorrelation 计算周期 T： 估计周期等于第一个过零点位置加上第一个峰值相对过零点的位置。
    T = zeroposi + max_position; % 周期 = 过零点后第一个峰值所处的滞后量

end

function ck = CK(x, T, M)

    % Calculate the Correlated Kurtosis

    if nargin < 3
        M = 2;
    end

    % Insure x is a row vector
    x = x(:)';

    N = length(x);
    x_shift = zeros(M + 1, N);

    x_shift(1, :) = x;

    for m = 1:M
        x_shift(m + 1, T + 1:end) = x_shift(m, 1:end - T); %把上一行的信号整体向右平移 T 个采样点，得到第 m+1 行作为延时信号。
    end

    ck = sum(prod(x_shift).^2) / sum(x.^2)^(M + 1); %将 x(t)*x(t-T)*x(t-2T)*... 逐点相乘
                                                    % sum(prod(...)^2) → 度量周期冲击程度（周期性越强，值越大）
end                                                 % sum(x.^2)^(M+1) → 归一化，避免能量大小影响对比

function [I, J, M] = max_IJ(X)
    % Returns the row and column indices of the maximum in matrix X.

    [temp, tempI] = max(X);
    [M, J] = max(temp);
    I = tempI(J);
end

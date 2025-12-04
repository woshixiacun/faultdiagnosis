
% 上海交通大学-包文杰 硕士论文
% 《加权谱峭度故障诊断方法研究与应用》
% 原始代码

%%  信号参数设置
clear;
fs = 10e3;                    % 采样频率
fn = 1125;                    % 周期性冲击信号共振频率
fn2 = 2250;                   % 调制干扰信号共振频率
a = 150;                      % 衰减系数 
A0 = 0.08;                    % 位移常数
B0 = 4 * A0;
SNR = -5;
C1 = 0.05;
C2 = 1;
f_mesh = 480;
f_shaft = 15;
B1 = pi/7;
B2 = pi/9;
T = 1/20;                    % 重复周期
N = 5*fs;                    % 采样点数
NT = round(fs*T);            % 单周期采样点数
tt0 = 0:1/fs:(NT-1)/fs;      % 单周期采样时刻
tt = 0:1/fs:(N-1)/fs;        % 采样时刻
p1 = floor(N/NT);            % 重复次数

%%  周期性冲击信号
s1 = [];

for i = 1:p1                               %产生p1个相同波形
    s1 = [s1 (A0.*exp(-a.*(tt0)).*cos(2*pi*fn*(tt0)))]; 
end
d = ((N/NT-p1)*NT);                %p1周期后剩下的采样点数
ttt0 = 0:1/fs:((N/NT-p1)*NT)/fs;   %p1周期后剩下的采样时刻
s1 = [s1 (A0.*exp(-a.*(ttt0)).*sin(2*pi*fn*(ttt0)))];
s1(1:NT) = 0;
s1 = s1(1: N);

%%  随机干扰冲击信号
s2_1 = (B0.*exp(-a.*(tt0)).*cos(2*pi*fn2*(tt0)));
s2 = [zeros(1, 2*fs) s2_1 zeros(1, 4*fs-2*fs-length(s2_1)) s2_1 zeros(1, N-4*fs-length(s2_1))];

%%  调制干扰信号
s3 = C1*sin(2*pi*f_mesh*tt+B1).*(1+C2*sin(2*pi*f_shaft*tt+B2));

%%  随机噪声
NOISE = randn(size(s1));
NOISE = NOISE-mean(NOISE);
signal_power = 1/length(s1)*sum(s1.*s1);
noise_variance = signal_power/(10^(SNR/10));
NOISE = sqrt(noise_variance)/std(NOISE)*NOISE;

%%  合成信号
s = s1 + s2 + s3 + NOISE;
save('./Bao_Sim.mat', 's');

%%  信号的时域图
figure(1)
subplot(511)
plot(tt,s1)         %原信号图像
axis([0,1,-0.1,0.1])
title('周期性冲击信号时域波形图')
subplot(512)
plot(tt(1:length(s2)),s2)
axis([1.5,4.5,-0.4,0.4])
title('随机干扰冲击信号时域波形图')
subplot(513)
plot(tt,s3)
axis([0,0.3,-0.15,0.15])
title('调制干扰信号时域波形图')
subplot(514)
plot(tt,NOISE)
title('随机噪声时域波形图')
subplot(515)
plot(tt,s)
title('合成信号时域波形图')

%%  信号的频域图
figure(2)
subplot(511)
N_s1 = length(s1);
HV = abs(hilbert(s1));
HV_m = HV - mean(HV);
HV_f = abs(fft(HV_m))/(N/2);
f = (0:N_s1-1)*fs/N_s1;
plot(f(1:fs), HV_f(1:fs))
title('周期性冲击信号包络谱');
subplot(512)
N_s2 = length(s2);
HV = abs(hilbert(s2));
HV_m = HV - mean(HV);
HV_f = abs(fft(HV_m))/(N/2);
f = (0:N_s2-1)*fs/N_s2;
plot(f(1:fs), HV_f(1:fs))
title('随机干扰冲击信号包络谱');
subplot(513)
N_s3 = length(s3);
HV = abs(hilbert(s3));
HV_m = HV - mean(HV);
HV_f = abs(fft(HV_m))/(N/2);
f = (0:N_s3-1)*fs/N_s3;
plot(f(1:fs), HV_f(1:fs))
title('调制干扰信号包络谱');
subplot(514)
N_n = length(NOISE);
HV = abs(hilbert(NOISE));
HV_m = HV - mean(HV);
HV_f = abs(fft(HV_m))/(N/2);
f = (0:N_n-1)*fs/N_n;
plot(f(1:fs), HV_f(1:fs))
title('随机噪声包络谱');
subplot(515)
N_s = length(s);
HV = abs(hilbert(s));
HV_m = HV - mean(HV);
HV_f = abs(fft(HV_m))/(N/2);
f = (0:N_s-1)*fs/N_s;
plot(f(1:fs), HV_f(1:fs))
title('合成信号包络谱');

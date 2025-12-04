
clc; clear; close all;


%%  simulated signal
%  参数设置
fs = 20000;                  % 采样频率
fn = 2000;                   % 周期性冲击信号共振频率
fn2 = 3600;
fn3 = 4800;
f_rand = 5100;                  % 随机干扰信号共振频率
a1 = 200;                    % 周期冲击衰减系数 
a2 = 1000;                   % 随机干扰衰减系数
A0 = 1;                      % 位移常数/幅值
B0 = 5 * A0;
SNR = -10;
C1 = 0.025;
C2 = 0.025;
f1 = 7;
f2 = 14;
B1 = pi/6;
B2 = -pi/3;
T = 1/29;                    % 重复周期

N = 1*fs;                    % 采样点数
NT = round(fs*T);            % 单周期采样点数
tt0 = 0:1/fs:(NT-1)/fs;      % 单周期采样时刻
t = 0:1/fs:(N-1)/fs;        % 采样时刻
t = t';
p1 = floor(N/NT);            % 重复次数

%% 周期性冲击信号
s1 = [];

for i = 1:p1                               %产生p1个相同波形
    s1 = [s1 (A0.*exp(-a1.*(tt0)).*cos(2*pi*fn*(tt0)))]; 
end
d = ((N/NT-p1)*NT);                %p1周期后剩下的采样点数
ttt0 = 0:1/fs:((N/NT-p1)*NT)/fs;   %p1周期后剩下的采样时刻
s1 = [s1 (A0.*exp(-a1.*(ttt0)).*sin(2*pi*fn*(ttt0)))];
s1(1:NT) = 0;
s1 = s1(1: N);
s1 = s1';

s11 = [];

for i = 1:p1                               %产生p1个相同波形
    s11 = [s11 (A0.*exp(-a1.*(tt0)).*cos(2*pi*fn2*(tt0)))]; 
end
d = ((N/NT-p1)*NT);                %p1周期后剩下的采样点数
ttt0 = 0:1/fs:((N/NT-p1)*NT)/fs;   %p1周期后剩下的采样时刻
s11 = [s11 (A0.*exp(-a1.*(ttt0)).*sin(2*pi*fn2*(ttt0)))];
s11(1:NT) = 0;
s11 = s11(1: N);
s11 = s11';

s111 = [];

for i = 1:p1                               %产生p1个相同波形
    s111 = [s111 (A0.*exp(-a1.*(tt0)).*cos(2*pi*fn3*(tt0)))]; 
end
d = ((N/NT-p1)*NT);                %p1周期后剩下的采样点数
ttt0 = 0:1/fs:((N/NT-p1)*NT)/fs;   %p1周期后剩下的采样时刻
s111 = [s111 (A0.*exp(-a1.*(ttt0)).*sin(2*pi*fn3*(ttt0)))];
s111(1:NT) = 0;
s111 = s111(1: N);
s111 = s111';

%% 随机干扰冲击信号
s2_1 = (B0.*exp(-a2.*(tt0)).*cos(2*pi*f_rand*(tt0)));
% s2 = [zeros(1, 2*fs) s2_1 zeros(1, 4*fs-2*fs-length(s2_1)) s2_1 zeros(1, N-4*fs-length(s2_1))];
s2 = [zeros(1, 0.49*fs),s2_1,zeros(1, N-0.49*fs-length(s2_1))];
s2 = s2';

%% 调制干扰信号
% s3 = C1*sin(2*pi*f1*t+B1).*(1+C2*sin(2*pi*f2*t+B2)); 
s3 = C1*sin(2*pi*f1*t+B1)+ C2*sin(2*pi*f2*t+B2);

% 无噪声信号
% sf = s1 + s2 + s3;
sf = s1 + s11 + s111 + s2 + s3;
%% 随机噪声
NOISE = randn(size(sf));
NOISE = NOISE-mean(NOISE);
signal_power = 1/length(sf)*sum(sf.*sf);
noise_variance = signal_power/(10^(SNR/10));
NOISE = sqrt(noise_variance)/std(NOISE)*NOISE;

%% 合成信号
s = sf + NOISE;

%% 
figure('Name', 'Time waveform of mixed signal with no noise')
plot(t, sf, 'b');
xlabel('Time [s]');
ylabel('Amplitude');

figure('Name', 'FFT amplitude spectrum of mixed signal with no noise')
[~, ~] = myfft(fs, sf, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');

figure('Name', 'Time waveform of mixed signal')
plot(t, s, 'b');
xlabel('Time [s]');
ylabel('Amplitude');

figure('Name', 'FFT amplitude spectrum of mixed signal')
[~, ~] = myfft(fs, s, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');

%% add noise
channel_1 = addnoise(sf,-8); 
channel_2 = addnoise(sf,-7);
channel_3 = addnoise(sf,-6);
channel_4 = addnoise(sf,-5);
channel_5 = addnoise(sf,-4);
channel_6 = addnoise(sf,-3);
channel_7 = addnoise(sf,-2);
channel_8 = addnoise(sf,-1);
channel_9 = addnoise(sf,0);
channel_10 = addnoise(sf,1);

noise_signal = [channel_1,channel_2,channel_3,channel_4,channel_5,channel_6,...
                channel_7,channel_8,channel_9,channel_10];

subplot(5,2,1)
plot(t,channel_1,'b')
subplot(5,2,2)
plot(t,channel_2,'b')
subplot(5,2,3)
plot(t,channel_3,'b')
subplot(5,2,4)
plot(t,channel_4,'b')
subplot(5,2,5)
plot(t,channel_5,'b')
subplot(5,2,6)
plot(t,channel_6,'b')
subplot(5,2,7)
plot(t,channel_7,'b')
subplot(5,2,8)
plot(t,channel_8,'b')
subplot(5,2,9)
plot(t,channel_9,'b')
subplot(5,2,10)
plot(t,channel_10,'b')

%%
figure('Name', 'Time-frequency waveform of mixed signal with different noise')
subplot(3,3,1)
plot(t, channel_1, 'b');
subplot(3,3,2)
[ff, amp] = myfft(fs, channel_1, 0);
plot(ff, amp / max(amp), 'b');
xlabel('Frequency [Hz]');
ylabel('Amplitude');
subplot(3,3,3)
envelope = abs(hilbert(channel_1)) - mean(abs(hilbert(channel_1)));
[ff, amp] = myfft(fs, envelope, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');
xlim([0 300])

subplot(3,3,4)
plot(t, channel_5, 'b');
subplot(3,3,5)
[ff, amp] = myfft(fs, channel_5, 0);
plot(ff, amp / max(amp), 'b');
xlabel('Frequency [Hz]');
ylabel('Amplitude');
subplot(3,3,6)
envelope = abs(hilbert(channel_5)) - mean(abs(hilbert(channel_5)));
[ff, amp] = myfft(fs, envelope, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');
xlim([0 300])

subplot(3,3,7)
plot(t, channel_10, 'b');
subplot(3,3,8)
[ff, amp] = myfft(fs,channel_10, 0);
plot(ff, amp / max(amp), 'b');
xlabel('Frequency [Hz]');
ylabel('Amplitude');
subplot(3,3,9)
envelope = abs(hilbert(channel_10)) - mean(abs(hilbert(channel_10)));
[ff, amp] = myfft(fs, envelope, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');
xlim([0 300])

%% VMD
alpha =2881;        % moderate bandwidth constraint
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
K =2;              % 4 modes
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 0.001;

[y_final, ~, ~] = VMD(s, alpha, tau, K, DC, init, tol);
y_final = y_final';
%% FMD_Parameters
filtersize = 30; %原始参数设置30/7/2/20
cutnum = 7;
modenum = 1;
maxiternum = 20;

% FMD 
% ！！！FMD的输入信号要求是列信号！！！！！！
tic
y_final = FMD(fs, s, filtersize, cutnum, modenum, maxiternum);
toc

%% Figure
set(0,'defaultfigurecolor','w')
b = size(y_final, 2);
figure('Name', 'Time waveform of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    plot(t, y_final(:, k), 'b')
    xlabel('Time [s]');
    ylabel('Amplitude');
end

figure('Name', 'FFT amplitude spectrum of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    [ff, amp] = myfft(fs, y_final(:, k), 0);
    plot(ff, amp / max(amp), 'r');
    xlabel('Frequency [Hz]');
    ylabel('Amplitude');
end

figure('Name', 'Hilbert envelope spectrum of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    envelope = abs(hilbert(y_final(:, k))) - mean(abs(hilbert(y_final(:, k))));
    [ff, amp] = myfft(fs, envelope, 1);
    xlabel('Frequency [Hz]');
    ylabel('Amplitude');
    xlim([0 300])
end


%% multi-channel

filtersize = 30; %原始参数设置30/7/2/20
cutnum = 7;
modenum = 1;
maxiternum = 20;

for i = 1:10
    ns = noise_signal(:,i);
    y = FMD(fs, ns, filtersize, cutnum, modenum, maxiternum);
    y_final(:,i) = y;
end

dims = 1; %文中设置为2
k = 13; %文中设置为Nm+3
% cm = ltsa(y_final, dims, k); %报错：第一个输入矩阵不是奇异矩阵
Y = lle(y_final,dims,k);

%%
set(0,'defaultfigurecolor','w')
b = size(y, 2);
figure('Name', 'Time waveform of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    plot(t, y(:, k), 'b')
    xlabel('Time [s]');
    ylabel('Amplitude');
end

figure('Name', 'FFT amplitude spectrum of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    [ff, amp] = myfft(fs, y(:, k), 0);
    plot(ff, amp / max(amp), 'r');
    xlabel('Frequency [Hz]');
    ylabel('Amplitude');
end

figure('Name', 'Hilbert envelope spectrum of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    envelope = abs(hilbert(y(:, k))) - mean(abs(hilbert(y(:, k))));
    [ff, amp] = myfft(fs, envelope, 1);
    xlabel('Frequency [Hz]');
    ylabel('Amplitude');
    xlim([0 300])
end

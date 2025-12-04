clc;
clear;
close all;

%%Dataset_1 3/4裂纹 转速600rpm 负载10Nm

load('E:\研究生资料\实验\数据\Dataset_1\LW-03\600-10.txt')
fs = 5000;

% 9列数据依次是位于实验台基座、减速箱输入端轴承、输出端轴承的三个传感器x/y/z三个方向的信号
% 重点关注第二个传感器（channel_4/5/6）的信号
channel_1 = X600_10(1:20000,1);
channel_2 = X600_10(1:20000,2);
channel_3 = X600_10(1:20000,3);
channel_4 = X600_10(1:20000,4);
channel_5 = X600_10(1:20000,5);
channel_6 = X600_10(1:20000,6);
channel_7 = X600_10(1:20000,7);
channel_8 = X600_10(1:20000,8);
channel_9 = X600_10(1:20000,9);

t = 1/fs:1/fs:4; t = t';

%%
set(0,'defaultfigurecolor','w')
subplot(3,1,1)
plot(t,channel_5,'b');
xlabel('Time [s]');
ylabel('Amplitude');
title('Time waveform of channel5')

subplot(3,1,2)
[ff, amp] = myfft(fs,channel_5,1);
plot(ff, amp / max(amp), 'r');
xlabel('Frequency [Hz]');
ylabel('Amplitude');
title('FFT amplitude spectrum of channel5')

subplot(3,1,3)
envelope = abs(hilbert(channel_5)) - mean(abs(hilbert(channel_5)));
[ff, amp] = myfft(fs, envelope, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');
xlim([0 300])
title('Hilbert envelope spectrum of channel5')


%% VMD
alpha =2881;        % moderate bandwidth constraint
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
K = 2;              % 4 modes
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 0.001;

[y_final, ~, ~] = VMD(channel_5, alpha, tau, K, DC, init, tol);
y_final = y_final';

b = size(y_final, 2);
set(0,'defaultfigurecolor','w')
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

%% FMD
filtersize = 30; %原始参数设置30/7/1/20
cutnum = 7;
modenum = 1;
maxiternum = 20;

tic
y_final = FMD(fs, channel_5, filtersize, cutnum, modenum, maxiternum);
toc

b = size(y_final, 2);
set(0,'defaultfigurecolor','w')
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

for i = 1:9
    x = X600_10(1:20000,i);
    y_final(:,i) = FMD(fs, x, filtersize, cutnum, modenum, maxiternum);
end

dims = 1; %文中设置为2
k = 13; %文中设置为Nm+3
% cm = ltsa(y_final, dims, k); %报错：第一个输入矩阵不是奇异矩阵
Y = lle(y_final,dims,k);

set(0,'defaultfigurecolor','w')
figure('Name', 'Time waveform of mode(s)')
plot(t, Y, 'b')
xlabel('Time [s]');
ylabel('Amplitude');


figure('Name', 'FFT amplitude spectrum of mode(s)')
[ff, amp] = myfft(fs,Y, 0);
plot(ff, amp / max(amp), 'r');
xlabel('Frequency [Hz]');
ylabel('Amplitude');


figure('Name', 'Hilbert envelope spectrum of mode(s)')
envelope = abs(hilbert(Y)) - mean(abs(hilbert(Y)));
[ff, amp] = myfft(fs, envelope, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');
xlim([0 300])




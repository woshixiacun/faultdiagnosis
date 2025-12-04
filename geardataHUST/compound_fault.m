
clc; clear; close all;

%%
load x
fs = 2e4;
t = (0:length(x) - 1)' / fs;

%% add noise
channel_1 = awgn(x,-5);
channel_2 = awgn(x,-4);
channel_3 = awgn(x,-3);
channel_4 = awgn(x,-2);
channel_5 = awgn(x,-1);
channel_6 = awgn(x,0);
channel_7 = awgn(x,1);
channel_8 = awgn(x,2);
channel_9 = awgn(x,3);
channel_10 = awgn(x,4);

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
modenum = 2;
maxiternum = 20;

% FMD
tic
y_final = FMD(fs, channel_6, filtersize, cutnum, modenum, maxiternum);
toc
% Plot


%% Figure
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
modenum = 2;
maxiternum = 20;

for i = 1:10
    ns = noise_signal(:,i);
    y = FMD(fs, ns, filtersize, cutnum, modenum, maxiternum);
    y_final(:,2*i-1:2*i) = y;
end

dims = 2; %文中设置为2
k = 13; %文中设置为Nm+3
% cm = ltsa(y_final, dims, k); %报错：第一个输入矩阵不是奇异矩阵
Y = lle(y_final,dims,k);

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

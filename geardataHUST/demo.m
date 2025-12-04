%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script illustrates the use of the FMD code "FMD.m"
%
%%%%%%%%%%%%%%%%
% Reference:
%%%%%%%%%%%%%%%%
%
%       Y. Miao, B. Zhang, C. Li, J. Lin, D. Zhang
%       "Feature Mode Decomposition:New Decomposition Theory
%       for Rotating Machinery Fault Diagnosis"
%       IEEE Transactions on Industrial Electronics.2022
%       DOI:10.1109/TIE.2022.3156156
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Code by Boyao Zhang
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%%
load x
fs = 2e4;
t = (0:length(x) - 1)' / fs;

%%
figure('Name', 'Time waveform of mixed signal')
plot(t, x, 'b');
xlabel('Time [s]');
ylabel('Amplitude');

figure('Name', 'FFT amplitude spectrum of mixed signal')
[~, ~] = myfft(fs, x, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');

% figure(3)
% hua_fft(x,fs,1);
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
y_final = FMD(fs, x, filtersize, cutnum, modenum, maxiternum);
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

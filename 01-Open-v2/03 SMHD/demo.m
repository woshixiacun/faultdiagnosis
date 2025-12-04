%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script illustrates the use of the smhd code "smhd.m"
%
%%%%%%%%%%%%%%%%
% Reference:
%%%%%%%%%%%%%%%%
%
%          ¡¾1¡¿Y. Miao, M. Zhao, J. Lin, Y. Lei
%           "Sparse maximum harmonics-to-noise-ratio deconvolution
%           for weak fault signature detection in bearings".
%       Measurement Science and Technology, 2016, 27(10)
%          ¡¾2¡¿Y. Miao, B. Zhang, J. Lin et al., 
%             ¡°A review on the application of blind deconvolution in machinery fault diagnosis¡±
%               Mechanical Systems and Signal Processing, 2022.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Code by Yonghao Miao
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
clc

%%
load sig3
x = x - mean(x);
addpath('..\00 subfunction\')

%%
fs = 20000;
N = length(x);
t = (0:N - 1) / fs;
t = t(:);
BPFI = 38;

%% Raw data
figure;
plot(t, x, 'b');
xlabel('Time [s]')
ylabel('Amplitude')
title('Raw data')
legend(['Kurtosis=', num2str(kurtosis(x))])
setfontsize(20);
set(gcf, 'position', [100, 100, 800, 400])
axis tight
ylim([-2 2.5])

envelope_x = abs(hilbert(x)) - mean(abs(hilbert(x)));
ff = 0:fs / N:fs - fs / N;
amp_envelope_x = abs(fft(envelope_x, N)) * 2 / fs;
figure;
plot(ff, amp_envelope_x, 'b')
xlabel('Frequency [Hz]')
ylabel('Amplitude')
setfontsize(20);
set(gcf, 'position', [100, 100, 800, 400])
axis tight
xlim([0, 200]);
ylim([0 0.025])

%% SMHD

[y_final, f_final, kurtIter] = smhd(fs, x, 100, 30, 1.5 * rms(x), [], 0);

%% Filtered signal
figure;
plot(t, y_final, 'b');
xlabel('Time [s]')
ylabel('Amplitude')
title('Filtered signal by SMHD')
legend(['Kurtosis=', num2str(kurtosis(y_final))])
setfontsize(20);
set(gcf, 'position', [100, 100, 800, 400])
axis tight
ylim([-3.5 4.5])

envelope_y = abs(hilbert(y_final)) - mean(abs(hilbert(y_final)));
amp_envelope_y = abs(fft(envelope_y, N)) * 2 / fs;
figure;
plot(ff, amp_envelope_y, 'b')
xlabel('Frequency [Hz]')
ylabel('Amplitude')
setfontsize(20);
set(gcf, 'position', [100, 100, 800, 400])
axis tight
xlim([0, 200]);
ylim([0 0.3])

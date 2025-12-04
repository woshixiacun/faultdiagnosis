%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script illustrates the use of the imckd code "imckd.m"
% In the field, we firstly proposed to estimate the iterative period by using the iterative algorithm to solve the problem of the prior period in blind deconvolution methods. 
%%%%%%%%%%%%%%%%
% Reference:
%%%%%%%%%%%%%%%%
% 
%                ¡¾1¡¿Y. Miao, M. Zhao, J. Lin, Y. Lei
%               "Application of an improved
%               maximum correlated kurtosis deconvolution method
%               for fault diagnosis of rolling element bearings"
%           Mechanical Systems and Signal Processing, 92 (2017) 173-195.
%                 ¡¾2¡¿Y. Miao, M. Zhao, K. Liang, J. Lin
%             ¡°Application of an improved MCKDA for fault detection 
%               of wind turbine gear based on encoder signal¡±
%                 Renewable Energy, 151 (2020) 192-203.
% 
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
load sig1
x = x - mean(x);
fs = 20000;             % Sampling frequency
addpath('..\00 subfunction\')

%% Raw data
N = length(x);
t = (0:N - 1) / fs;
figure;
plot(t, x, 'b');
xlabel('Time [s]')
ylabel('Amplitude')
title('Raw data')
legend(['Kurtosis=', num2str(kurtosis(x))])
setfontsize(20);
set(gcf, 'position', [100, 100, 800, 400])
axis tight

envelope_x = abs(hilbert(x)) - mean(abs(hilbert(x)));
ff = 0:fs / N:fs - fs / N;
envelope_spec_x = abs(fft(envelope_x, N)) * 2 / fs;
figure;
plot(ff, envelope_spec_x, 'b')
xlabel('Frequency [Hz]')
ylabel('Amplitude')
setfontsize(20);
set(gcf, 'position', [100, 100, 800, 400])
axis tight
xlim([0, 200]);
ylim([0 0.035])

%%
L = 50;                 % Filter size
T = [];                 % Period
M = 1;                  % Shaft order
 
%%
[y, f, ckIter] = imckd(fs, x, L, 30, T, M);

%% Filtered signal
figure;
plot(t, y, 'b');
xlabel('Time [s]')
ylabel('Amplitude')
title('Filtered signal by IMCKD')
legend(['Kurtosis=', num2str(kurtosis(y))])
setfontsize(20);
set(gcf, 'position', [100, 100, 800, 400])
axis tight

envelope_y = abs(hilbert(y)) - mean(abs(hilbert(y)));
envelope_spec_y = abs(fft(envelope_y, N)) * 2 / fs;
figure;
plot(ff, envelope_spec_y, 'b')
xlabel('Frequency [Hz]')
ylabel('Amplitude')
setfontsize(20);
set(gcf, 'position', [100, 100, 800, 400])
axis tight
xlim([0, 200]);
ylim([0 0.35])

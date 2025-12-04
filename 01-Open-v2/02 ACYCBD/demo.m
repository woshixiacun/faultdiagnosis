%--------------------------------------------------------------------------
%
% This script provides a simulated case of the ACYCBD code "ACYCBD.m"
%
%---------------
% Reference:
%---------------
%
%   Author: ¡¾1¡¿B. Zhang, Y. Miao, J. Lin, Y. Yi
%               "Adaptive maximum second-order cyclostationarity blind deconvolution
%               and its application for locomotive bearing fault diagnosis"
%           Mechanical Systems and Signal Processing, 158 (2021) 107736.
%          ¡¾2¡¿Y. Miao, B. Zhang, J. Lin et al., 
%             ¡°A review on the application of blind deconvolution in machinery fault diagnosis¡±
%               Mechanical Systems and Signal Processing, 2022.
%
%-------------------------------------------------
%
% Author: @ Zhang Boyao
%
%-------------------------------------------------
%--------------------------------------------------------------------------

clear
close all
clc

%% load simulated data
load sig2
x = x - mean(x);
addpath('..\00 subfunction\')

%% Plot raw data
% Raw data
figure
plot((0:length(x) - 1)' / fs, x, 'b')
set(get(gca, 'XLabel'), 'String', 'Time [s]');
set(get(gca, 'YLabel'), 'String', 'Amplitude');
set(get(gca, 'Title'), 'String', 'Raw data in time domain');
set(gcf, 'pos', [400 100 800 400])
setfontsize(18)

% Envelope spectrum of raw data
en_x = abs(hilbert(x));
en_spec_x = abs(fft(en_x - mean(en_x), length(en_x))) * 2 / length(en_x);
figure
plot(0:fs / length(x):fs - fs / length(x), en_spec_x, 'b')
set(get(gca, 'XLabel'), 'String', 'Frequency [Hz]');
set(get(gca, 'YLabel'), 'String', 'Amplitude');
set(get(gca, 'Title'), 'String', 'Envelope spectrum of Raw data');
set(gcf, 'pos', [400 100 800 400])
setfontsize(18)
xlim([0 200])
ylim([0 0.12]);

%% ACYCBD
[h_final, s, kappa, W, count, err, f_est] = ACYCBD(x, fs, 40);

%% Plot deconvolved result
% Estimated cyclic frequency
figure
plot(1:count, f_est, 'b-*', 'LineWidth', 1)
hold on
plot((1:count), BPFI * ones(count, 1), 'r-^', 'LineWidth', 1)
set(get(gca, 'XLabel'), 'String', 'Iteration');
set(get(gca, 'YLabel'), 'String', 'Frequency [Hz]');
set(get(gca, 'Title'), 'String', 'Estimated cyclic frequency');
set(gcf, 'pos', [400 100 800 400])
setfontsize(18)

% Deconvolved signal
figure
plot((0:length(s) - 1) / fs, s, 'b')
set(get(gca, 'XLabel'), 'String', 'Time [s]');
set(get(gca, 'YLabel'), 'String', 'Amplitude');
set(get(gca, 'Title'), 'String', 'Deconvolved signal in time domain by ACYCBD');
set(gcf, 'pos', [400 100 800 400])
setfontsize(18)

% Envelope spectrum of Deconvolved signal
s = s - mean(s);
en_s = abs(hilbert(s));
en_spec_s = abs(fft(en_s - mean(en_s), length(en_s))) * 2 / length(en_s);
figure
plot(0:fs / length(en_s):fs - fs / length(en_s), en_spec_s, 'b')
set(get(gca, 'XLabel'), 'String', 'Frequency [Hz]');
set(get(gca, 'YLabel'), 'String', 'Amplitude');
set(get(gca, 'Title'), 'String', 'Envelope spectrum of deconvolved signal by ACYCBD');
set(gcf, 'pos', [400 100 800 400])
setfontsize(18)
xlim([0 200])
ylim([0 0.3]);

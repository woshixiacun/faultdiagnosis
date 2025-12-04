function signal_noise = addnoise(x,SNR)

         NOISE = randn(size(x));
         NOISE = NOISE-mean(NOISE);
         signal_power = 1/length(x)*sum(x.*x);
         noise_variance = signal_power/(10^(SNR/10));
         signal_noise = sqrt(noise_variance)/std(NOISE)*NOISE;

end
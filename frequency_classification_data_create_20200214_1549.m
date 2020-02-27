close all;

max_samp_per_sec = 1.2*10^6;
n_f_types = 27;

f_s = max_samp_per_sec;
dt = 1/f_s;
f_max = max_samp_per_sec/100; % At least 100 samples per cycle

stop_time = 10/(f_max/27);    % 10 cycles of the lowest frequency 
t = (0:dt:stop_time-dt);     % seconds
packet_length = length(t);

n_packets_per_type = 2000;
dataset = zeros(n_f_types+1, n_packets_per_type, packet_length, 'uint16');
gain = 0.25;
apollo3_blue_adc_v_ref = 1.5;
adc_bit_resolution = 14;
offset = apollo3_blue_adc_v_ref/2;
snr = 20;

temp_dataset = zeros(n_packets_per_type, packet_length);
temp_dataset = abs(awgn(temp_dataset+1.0, snr, 'measured')-1.0); % Creating noise packets
dataset(1,:,:) = uint16(round((2^adc_bit_resolution-1)*...
        temp_dataset/apollo3_blue_adc_v_ref));
    
classifications = uint16(zeros(n_f_types+1,n_packets_per_type));

for f_divisor = 1:n_f_types
    f0 = f_max/f_divisor;
    orig_sig = gain*sin(2*pi*f0.*t)+ offset;
    temp_dataset = repmat(squeeze(orig_sig),n_packets_per_type,1);  
    temp_dataset = awgn(temp_dataset, snr, 'measured');
    dataset_idx = (n_f_types+1)-(f_divisor-1); %accounts for Noise in 0th idx
    dataset(dataset_idx,:,:) = uint16(round((2^adc_bit_resolution-1)*...
        temp_dataset/apollo3_blue_adc_v_ref));  
    if(f0 - 2^16 > 0)
        f0
        f_divisor
        break;
    end
    classifications(dataset_idx,:) = uint16(f0);

%     Filtering out the f0. Leaves only noise.
%     wo = f0/(f_s/2); bw = wo/10; %wo/35;
%     [b,a] = iirnotch(wo,bw);
%     sig_filt=filter(b,a,sig_plus_noise);
%     plot(t,sig_filt, 'r');
end

figure;
subplot(1,1,1)
for f_idx = 1:n_f_types+1
    plot(t,squeeze(dataset(f_idx,1,:)));
    hold on;   
end
xlabel('time (s)');
zoom xon;

% writematrix(dataset,'f_sweep_02.csv','Delimiter',',')  
% writematrix(classifications,'f_class_02.csv','Delimiter',',') 

save('f_sweep_05.mat', 'dataset', '-v7.3') % For variables larger than 2GB use MAT-file version 7.3 or later
% save('f_sweep_04.mat', 'dataset')
save('f_features_05.mat', 'classifications', '-v7.3')
% save('f_features_05.mat', 'classifications')

% fid = fopen('f_sweep_00.dat', 'w');
% fwrite(fid, dataset);
% fclose(fid);
% 
% fid = fopen('f_class_00.dat', 'w');
% fwrite(fid, classifications);
% fclose(fid);
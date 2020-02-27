import os
import math
import scipy.io as sio
import scipy.signal as ss
import numpy as np
import pickle
from tqdm import tnrange
import heapq

def load_mat_size(filename):
    '''
    This function loads in and processes matlab struct 
    '''

    dataDict = sio.loadmat(filename, squeeze_me = True, struct_as_record=False)
    dataStruct = dataDict['dataset']
    signal = dataStruct.complexBasebandSignal
        
    return np.size(signal, 0)

def scale_and_float16_data(data, n_largest=100, is_verbose = True):
    data_scaled = np.zeros(np.shape(data))
    for data_idx in range(np.size(data,0)):
        avg_max = np.mean(heapq.nlargest(n_largest, data[data_idx,:])) # result = np.argpartition(sig, -n_largest, axis=1)[:,-n_largest:]
        scaling = 1.0/avg_max # Scaling the max average to 1V
        data_scaled[data_idx,:] = data[data_idx,:]*scaling

        # https://gamedev.stackexchange.com/questions/28023/python-float-32bit-to-half-float-16bit
    data_scaled_quantized = data_scaled.astype(np.float16) # Convert float64 to float16
    return data_scaled_quantized

def scale_and_quantize_sig(sig, n_largest=100, scaling_target=1.0, \
        adc_resolution=14, adc_v_ref=2.0):
    sig_scaled = np.zeros(np.shape(sig))
    for sig_idx in range(np.size(sig,0)):
        avg_max = np.mean(heapq.nlargest(n_largest, sig[sig_idx,:])) # result = np.argpartition(sig, -n_largest, axis=1)[:,-n_largest:]
        scaling = scaling_target/avg_max # Scaling the max average to scaling_target volts
        sig_scaled[sig_idx,:] = sig[sig_idx,:]*scaling
        
    sig_scaled_quantized = quantize_sig(sig_scaled, adc_resolution, adc_v_ref) # !!! Assumes: y_intercept = 0 and ADC operation is equivalent of np.floor(...)
    
    if adc_resolution <= 8:
        return sig_scaled_quantized.astype(np.uint8)
    elif adc_resolution <= 16:
        return sig_scaled_quantized.astype(np.uint16)
    else:
        return sig_scaled_quantized.astype(np.uint32)

def quantize_sig(sig, adc_resolution=14, adc_v_ref=2.0):
    sig_quantized = np.floor((2**adc_resolution-1)/adc_v_ref*sig) # !!! Assumes: y_intercept = 0 and ADC operation is equivalent of np.floor(...)
    
    if adc_resolution <= 8:
        return sig_quantized.astype(np.uint8)
    elif adc_resolution <= 16:
        return sig_quantized.astype(np.uint16)
    else:
        return sig_quantized.astype(np.uint32)

def pickle_tensor(data, filename, is_verbose = True):
    '''
    Pickle the data
    '''
    outfile = open(filename,'wb')
    pickle.dump(data, outfile, protocol=4)
    outfile.close()
    
    if is_verbose == True:
        print("File saved as:" + filename)

    return 



def unpickle_file(filename, is_verbose = True):
    '''
    Unpickle the data
    '''
    fo = open(filename, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    
    if is_verbose == True:
        print("File loaded:" + filename)
        
    return data

def complex_baseband_rf_signal_to_envelope(protocol_idx, n_snr, center_freq, samp_rate,\
    data_packets, ctrl_packets, num_data_classes, num_ctrl_classes, packets_per_class,\
    samps_per_pckt, classification_type, address_idx = None):
    '''
    This function takes I and Q values of a signal and outputs a baseband envelope signal
    '''
    
    # Define timestep
    ts = 1/samp_rate
    T = samps_per_pckt*ts
    t = np.arange(0,T,ts)
    t = t[:samps_per_pckt]
    
    x_data = np.empty([num_data_classes+num_ctrl_classes, n_snr, packets_per_class, samps_per_pckt])
    y_data = np.empty([num_data_classes+num_ctrl_classes, n_snr, packets_per_class])
    
    #### Convert Complex Baseband to Enevelope ####
    
    # Data Packets
    for ii in range(num_data_classes+num_ctrl_classes):
        for mm in range(n_snr):
            for jj in range(packets_per_class):
                if ii < num_data_classes:
                    data_i = data_packets[ii, mm, jj, : ,0]
                    data_q = data_packets[ii, mm, jj, : ,1]
                elif ii >= num_data_classes:
                    data_i = ctrl_packets[ii-num_data_classes, mm, 0, : ,0]
                    data_q = ctrl_packets[ii-num_data_classes, mm, 0, : ,1]
        
                # Reconstruct Signal
                # signal = data_i*np.cos(2*np.pi*center_freq*t) - data_q*np.sin(2*np.pi*center_freq*t)
        
                # Generate Envelope
                env_s = np.abs(data_i+data_q)
            
                x_data[ii, mm, jj, :] = env_s
                if classification_type == 'protocol':
                    y_data[ii, mm, jj] = protocol_idx
                elif classification_type == 'packet':
                    if protocol_idx == 0: # Noise Signal
                        y_data[ii, mm, jj] = 0
                    else:
                        y_data[ii, mm, jj] = ii + 1 # Noise is the 0th packet classification
                elif classification_type == 'protocolAndPacket':
                    if protocol_idx == 0: # Noise Signal
                        y_data[ii, mm, jj] = 0
                    else:
                        y_data[ii, mm, jj] = (protocol_idx-1)*(num_data_classes+num_ctrl_classes) + ii + 1 # Noise is the 0th packet classification
                elif classification_type == 'address':
                    y_data[ii, mm, jj] = address_idx
    
    return np.array(x_data), np.array(y_data)

def split_tensor_without_shuffle(x_data, y_data, data_len_mat = None, data_rate_mat = None, validation_fraction=0.2):
    """
    Splits the data into training and validation data
    according to the fraction that was specified. The samples are shuffled and then selected.
    The data is equally splitted along classes and signal to noise ratios.
    The new data array, validation array and the according label arrays are returned.
    """
    # Split data
    nb_sets = x_data.shape[6]
    nb_cutted = int(np.floor(nb_sets * validation_fraction))

    x_test = x_data[:,:,:,:,:,:,-1:(-nb_cutted-1):-1,:]
    y_test = y_data[:,:,:,-1:(-nb_cutted-1):-1]
    x_data = np.delete(x_data, np.s_[-1:(-nb_cutted-1):-1], axis=6)
    y_data = np.delete(y_data, np.s_[-1:(-nb_cutted-1):-1], axis=3)

    if data_len_mat is not None:
        data_len_mat_test = data_len_mat[:,:,:,-1:(-nb_cutted-1):-1]
        data_len_mat = np.delete(data_len_mat, np.s_[-1:(-nb_cutted-1):-1], axis=3)

    if data_rate_mat is not None:
        data_rate_test = data_rate_mat[:,:,:,-1:(-nb_cutted-1):-1]
        data_rate_mat = np.delete(data_rate_mat, np.s_[-1:(-nb_cutted-1):-1], axis=3)

    return x_data, y_data, data_len_mat, data_rate_mat, x_test, y_test, data_len_mat_test, data_rate_test

def min_max_scale(data, range_to_scale=(0,1)):
    mi, ma = np.float32(range_to_scale)
    data = np.float32(np.float32(data - data.min()) / np.float32(data.max() - data.min())) #need to conserve memory
    data = np.float32(data * (ma - mi) + mi)
    return data

def convert_frequency_to_idx(frequency):
    """
    Converts class_idx to a frequency
    """
    frequency_idx = {
        0: 0,
        444: 1,
        462: 2,
        480: 3,
        500: 4,
        522: 5,
        545: 6,
        571: 7,
        600: 8,
        632: 9,
        667: 10,
        706: 11,
        750: 12,
        800: 13,
        857: 14,
        923: 15,
        1000: 16,
        1091: 17,
        1200: 18,
        1333: 19,
        1500: 20,
        1714: 21,
        2000: 22,
        2400: 23,
        3000: 24,
        4000: 25,
        6000: 26,
        12000: 27    
    }
    # frequency_idx = {
    #     'Noise': 0,
    #     '444Hz': 1,
    #     '462Hz': 2,
    #     '480Hz': 3,
    #     '500Hz': 4,
    #     '522Hz': 5,
    #     '545Hz': 6,
    #     '571Hz': 7,
    #     '600Hz': 8,
    #     '632Hz': 9,
    #     '667Hz': 10,
    #     '706Hz': 11,
    #     '750Hz': 12,
    #     '800Hz': 13,
    #     '857Hz': 14,
    #     '923Hz': 15,
    #     '1000Hz': 16,
    #     '1091Hz': 17,
    #     '1200Hz': 18,
    #     '1333Hz': 19,
    #     '1500Hz': 20,
    #     '1714Hz': 21,
    #     '2000Hz': 22,
    #     '2400Hz': 23,
    #     '3000Hz': 24,
    #     '4000Hz': 25,
    #     '6000Hz': 26,
    #     '12000Hz': 27    
    # }

    return frequency_idx[frequency]

def convert_idx_to_frequency(class_idx):
    """
    Converts class_idx to a frequency
    """
    frequency_label = {
        0: 'Noise',
        1: '444Hz',
        2: '462Hz',
        3: '480Hz',
        4: '500Hz',
        5: '522Hz',
        6: '545Hz',
        7: '571Hz',
        8: '600Hz',
        9: '632Hz',
        10: '667Hz',
        11: '706Hz',
        12: '750Hz',
        13: '800Hz',
        14: '857Hz',
        15: '923Hz',
        16: '1000Hz',
        17: '1091Hz',
        18: '1200Hz',
        19: '1333Hz',
        20: '1500Hz',
        21: '1714Hz',
        22: '2000Hz',
        23: '2400Hz',
        24: '3000Hz',
        25: '4000Hz',
        26: '6000Hz',
        27: '12000Hz'    
    }

    # print('class_idx = {}| label={}'.format(class_idx, frequency_label[class_idx]))

    return frequency_label[class_idx]

# if files exist in the folder, delete them
def delete_files_in_folder(folder):
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path) # deletes the file path
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

# ranomly permute the matrices in unison
def unison_shuffled_copies(a, b, c = None, d = None):
    p = np.random.permutation(len(a))
    if d is not None:
        assert len(a) == len(b) == len(c) == len(d)
        return a[p], b[p], c[p], d[p]
    if c is not None:
        assert len(a) == len(b) == len(c)
        return a[p], b[p], c[p]
    else:
        assert len(a) == len(b)
        return a[p], b[p]

def decimate(signal, dec_factor):
    return ss.decimate(signal, dec_factor)


def getFFT(signal):
    return 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(signal))))

def convert_bytes(file_bytes):
    for units in ['B', 'KB', 'MB', 'GB', 'TB']:
        if file_bytes < 1024.0:
            return file_bytes, units
        file_bytes /= 1024.0

def tflite_conversion(tf, model_final_save_path, model_tflite_path, model_tflite_quant_path):
    # import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_final_save_path)

    # TFLite Version
    tflite_model = converter.convert()    
    with open(model_tflite_path, 'wb') as f:
      f.write(tflite_model)

    file_bytes_tflite = int(os.path.getsize(model_tflite_path))
    file_size, units = convert_bytes(file_bytes_tflite)
    print('\nTFLite File Size: {:.3f}{}'.format(file_size, units)) 

    # Quantized TFLite Version
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    with open(model_tflite_quant_path, 'wb') as f:
      f.write(tflite_quant_model)

    file_bytes_tflite_quant = int(os.path.getsize(model_tflite_quant_path))
    file_size, units = convert_bytes(file_bytes_tflite_quant)
    print('\nTFLite Quantized File Size: {:.3f}{}'.format(file_size, units))

    return file_bytes_tflite, file_bytes_tflite_quant
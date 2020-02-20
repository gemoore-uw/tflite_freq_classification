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

def load_mat_file_ri(filename, numDataPackets, numCtrlClasses, output_size = False):
    '''
    This function loads in and processes matlab struct 
    '''

    dataDict = sio.loadmat(filename, squeeze_me = True, struct_as_record=False)
    dataStruct = dataDict['dataset']
    if output_size is True:
        signal = dataStruct.complexBasebandSignal
        return
    snr = dataStruct.snr
    packetType = dataStruct.packetType
    packetNum = dataStruct.packetNum
    numSamples = dataStruct.numSamples
    signal = dataStruct.complexBasebandSignal

    #### Split data packets and control packets ####
    dataPckts = signal[:,0:numDataPackets]
    ctrlPckts = signal[:,numDataPackets:numDataPackets+numCtrlClasses]

    #### Seperate I and Q values ####
    dataPckts_r = np.real(dataPckts)
    dataPckts_i = np.imag(dataPckts)

    ctrlPckts_r = np.real(ctrlPckts)
    ctrlPckts_i = np.imag(ctrlPckts)

    print("dataPckts_r Shape: {}".format(np.shape(dataPckts_r)))
    print("ctrlPckts_r Shape: {}".format(np.shape(ctrlPckts_r)))
    
    return dataPckts_r, dataPckts_i, ctrlPckts_r, ctrlPckts_i

def load_mat_file_ri(filename, numDataPackets, numCtrlClasses, output_size = False):
    '''
    This function loads in and processes matlab struct 
    '''

    dataDict = sio.loadmat(filename, squeeze_me = True, struct_as_record=False)
    dataStruct = dataDict['dataset']
    if output_size is True:
        signal = dataStruct.complexBasebandSignal
        return
    snr = dataStruct.snr
    packetType = dataStruct.packetType
    packetNum = dataStruct.packetNum
    numSamples = dataStruct.numSamples
    signal = dataStruct.complexBasebandSignal

    #### Split data packets and control packets ####
    dataPckts = signal[:,0:numDataPackets]
    ctrlPckts = signal[:,numDataPackets:numDataPackets+numCtrlClasses]

    #### Seperate I and Q values ####
    dataPckts_r = np.real(dataPckts)
    dataPckts_i = np.imag(dataPckts)

    ctrlPckts_r = np.real(ctrlPckts)
    ctrlPckts_i = np.imag(ctrlPckts)

    print("dataPckts_r Shape: {}".format(np.shape(dataPckts_r)))
    print("ctrlPckts_r Shape: {}".format(np.shape(ctrlPckts_r)))
    
    return dataPckts_r, dataPckts_i, ctrlPckts_r, ctrlPckts_i

    '''
    This function takes the data and creates a multidimensional array
    '''
def process_data_ri(totalPackets, packetsPerClass, numClasses, n_snr, \
                            samplesPerPacket, sampleType, real_data, imag_data):

    data = np.empty([numClasses, n_snr, packetsPerClass, samplesPerPacket, sampleType])
    for class_ in range(numClasses):
        for snr in range(n_snr):
            for packetNum in range(packetsPerClass):
                data[class_, snr, packetNum, :, 0] = real_data[:, class_*packetsPerClass + packetNum]
                data[class_, snr, packetNum, :, 1] = imag_data[:, class_*packetsPerClass + packetNum]
    return data

def process_sig_env(protocol, packet_type, env_sig,\
        center_freq, snr, downsample, mcs, ht):

    data[protocol, packet_type, :,:] = env_sig
    # data_features[protocol, packet_type,:] = [center_freq, snr, downsample, mcs, ht]
    # if packet_type_of_4 == 'Control':

    # else: # packet_type = A Data Class
    #     for protocol_ in range(n_protocol):
    #         for packet_type_ in range(n_packet_type):
    #             for packet_ in range(n_packets_per_type)
    #                 data[protocol_, packet_type_, packet_,:] = env_sig[packet_, :]
    #                 data_features[]



    #             for center_freq_ in range(n_center_freq):
    #                 for snr_ in range(n_snr):
    #                     for downsample_ in range(n_downsample):
    #                         for mcs_ in range(n_mcs):
    #                             for ht_ in range(n_ht):
    #     env_sig

    #     for class_ in range(numClasses):
    #         for snr in range(n_snr):
    #             for packetNum in range(packetsPerClass):
    #                 data[class_, snr, packetNum, :, 0] = real_data[:, class_*packetsPerClass + packetNum]
    #                 data[class_, snr, packetNum, :, 1] = imag_data[:, class_*packetsPerClass + packetNum]
    return data

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
    
def convert_idx_to_packet_name(class_idx):
    """
    Converts class_idx to packet_type name
    """
    packet_names = {
        0: 'Noise',
        1: 'Beacon',
        2: 'Data',
        3: 'QoS Data',
        4: 'ACK',
        5: 'RTS',
        6: 'CTS',
        7: 'Null',
        8: 'QoS Null',
        9: 'Blk ACK'
    }
    return packet_names[class_idx]

def convert_idx_to_rf_signal_name(rf_signal_list, class_idx):
    """
    Converts class_idx to protocol name
    """
    if len(rf_signal_list) == 4:
        protocol_names = {
            0: 'Noise',
            1: '802.11b',
            2: '802.11g',
            3: '802.11n-HT'        
        }
    elif len(rf_signal_list) == 3: # removing 'b' protocol from dataset
        protocol_names = {
            0: 'Noise',
            1: '802.11g',
            2: '802.11n-HT'        
        }

    return protocol_names[class_idx]

def convert_idx_to_protocol_and_packet_name(rf_signal_list, class_idx):
    """
    Converts class_idx to protocol + packet_type name
    """
    if len(rf_signal_list) == 4:
        protocol_and_packet_names = {
            0: 'Noise',
            1: '802.11b Beacon',
            2: '802.11b Data',
            3: '802.11b QoS Data',
            4: '802.11b ACK',
            5: '802.11b RTS',
            6: '802.11b CTS',
            7: '802.11b Null',
            8: '802.11b QoS Null',
            9: '802.11b Blk ACK',
            10: '802.11g Beacon',
            11: '802.11g Data',
            12: '802.11g QoS Data',
            13: '802.11g ACK',
            14: '802.11g RTS',
            15: '802.11g CTS',
            16: '802.11g Null',
            17: '802.11g QoS Null',
            18: '802.11g Blk ACK',
            19: '802.11n-HT Beacon',
            20: '802.11n-HT Data',
            21: '802.11n-HT QoS Data',
            22: '802.11n-HT ACK',
            23: '802.11n-HT RTS',
            24: '802.11n-HT CTS',
            25: '802.11n-HT Null',
            26: '802.11n-HT QoS Null',
            27: '802.11n-HT Blk ACK'    
        }
    elif len(rf_signal_list) == 3: # removing 'b' protocol from dataset
        protocol_and_packet_names = {
            0: 'Noise',
            1: '802.11g Beacon',
            2: '802.11g Data',
            3: '802.11g QoS Data',
            4: '802.11g ACK',
            5: '802.11g RTS',
            6: '802.11g CTS',
            7: '802.11g Null',
            8: '802.11g QoS Null',
            9: '802.11g Blk ACK',
            10: '802.11n-HT Beacon',
            11: '802.11n-HT Data',
            12: '802.11n-HT QoS Data',
            13: '802.11n-HT ACK',
            14: '802.11n-HT RTS',
            15: '802.11n-HT CTS',
            16: '802.11n-HT Null',
            17: '802.11n-HT QoS Null',
            18: '802.11n-HT Blk ACK'        
        }

    return protocol_and_packet_names[class_idx]

def convert_idx_to_address(rf_signal_list, class_idx):
    return class_idx

def convert_std_and_packet_type_to_idx(rf_signal_list, std_idx, packet_type_idx):
    if std_idx == 0:
        return 0
    else:
        if len(rf_signal_list) == 4:
            return 9*(std_idx-1)+packet_type_idx + 1
        elif len(rf_signal_list) == 3:
            return 9*(std_idx-2)+packet_type_idx + 1
        else:
            raise Exception('[{}] is not a valid rf_signal_list'\
            .format(len(rf_signal_list)))

def convert_std_to_idx(std_802p11):
    """
    Converts std_802p11 entry to an index within the associated array
    """
    if std_802p11 == 'x':
        idx = 0
    elif std_802p11 == 'b':
        idx = 1
    elif std_802p11 == 'g':
        idx = 2
    elif std_802p11 == 'n':
        idx = 3
    else:
        raise Exception('[{}] is not a valid std_802p11'\
            .format(std_802p11))
    return idx

def convert_packet_type_to_idx(packet_type):
    """
    Converts std_802p11 entry to an index within the associated array
    """
    if packet_type < 9:
        idx = packet_type
    else:
        raise Exception('[{}] is not a valid packet_type'\
            .format(packet_type))
    return int(idx)

def convert_center_freq_to_idx(center_freq):
    """
    Converts center_freq entry to an index within the associated array
    """

    if center_freq == 2412:
        idx = 0
    else:
        raise Exception('[{}] is not a valid center_freq'\
            .format(center_freq))
    return idx

def convert_snr_to_idx(snr):
    """
    Converts snr entry to an index within the associated array
    """

    if snr == 15:
        idx = 0
    else:
        raise Exception('[{}] is not a valid snr'\
            .format(snr))
    return idx

def convert_downsample_to_idx(downsample_ratio):
    """
    Converts downsample_ratio entry to an index within the associated array
    """

    if downsample_ratio == 1 or downsample_ratio == 0: #!!! Remove the 0 condition once the datasets have been fixed
        idx = 0
    elif downsample_ratio == 2.0:
        idx = 1
    elif downsample_ratio == 4.0:
        idx = 2
    else:
        raise Exception('[{}] is not a valid downsample_ratio'\
            .format(downsample_ratio))
    return idx

def convert_downsample_idx_to_n(idx):
    """
    Converts downsample_ratio entry to an index within the associated array
    """

    if idx == 0: #!!! Remove the 0 condition once the datasets have been fixed
        downsample_ratio = 1
    elif idx == 1:
        downsample_ratio = 2
    elif idx == 2:
        downsample_ratio = 4
    else:
        raise Exception('[{}] is not a valid idx'\
            .format(idx))
    return downsample_ratio

def convert_mcs_and_data_rate_to_idx(mcs, data_rate=1):
    """
    Converts mcs entry to an index within the associated array
    """

    if mcs =='X':
        if data_rate == 1:
            idx = 0
        elif data_rate == 2:
            idx = 1
        elif data_rate == 5.5:
            idx = 2
        elif data_rate == 11:
            idx = 3
        else:
            raise Exception('[{}] is not a valid data_rate'\
            .format(data_rate))
    elif float(mcs) < 8:
        idx = float(mcs)
    else:
        raise Exception('[{}] is not a valid MCS'\
            .format(mcs))

    return int(idx)

def convert_ht_to_idx(ht):
    """
    Converts ht entry to an index within the associated array
    """
    if ht == 0:
        idx = 0
    elif ht == 1:
        idx = 1
    else:
        raise Exception('[{}] is not a valid HT'\
            .format(ht))
    return idx

def address_name(address_list, class_idx):
    assert (class_idx == 99), 'Not a valid Address'
    address_name = 'an address'

    return address_name

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
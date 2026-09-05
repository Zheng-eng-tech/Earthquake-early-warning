function [FV, debug_info] = st_FV_final(st, pb_inst, pzfile, fmin, fmax, debug_prefix)
%ST_FV Corrected MATLAB one-to-one implementation of Python st_FV.
%
%   FV = st_FV(st, pb_inst, pzfile, fmin, fmax)
%   [FV, debug_info] = st_FV(st, pb_inst, pzfile, fmin, fmax, debug_prefix)
%
%   This version is rebuilt from the verified MATLAB debug stages:
%   - ObsPy cosine taper exact translation
%   - ObsPy Butterworth SOS filter with zerophase=True
%   - Python-style Welch PSD
%   - Verified 7 time features, 10 envelope features, 15 spectrum features,
%     and 13 python_speech_features-style MFCC features
%
%   NOTE:
%   This version includes a MATLAB PAZ deconvolution branch for pb_inst=true.
%   It follows the Python logic: read SAC Pole-Zero file, remove two zero
%   zeros, use PAZ gain as constant, then deconvolve the PAZ response.

    if nargin < 2 || isempty(pb_inst)
        pb_inst = false;
    end
    if nargin < 3
        pzfile = [];
    end
    if nargin < 4 || isempty(fmin)
        fmin = 1.0;
    end
    if nargin < 5 || isempty(fmax)
        fmax = 7.0;
    end
    if nargin < 6
        debug_prefix = '';
    end

    save_debug = ~isempty(debug_prefix);

    FV = [];
    debug_info = struct();

    st_cp = st;
    Fs = double(st_cp(1).SampleRate);

    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage00_input.csv'], 'stage00_input');
    end

    %% =========================================================
    %  First signal conditioning
    %  Python:
    %    st_cp.detrend('demean')
    %    st_cp.detrend('linear')
    %    st_cp.taper(0.05, 'cosine', side='both')
    %    st_cp.filter(..., corners=4, zerophase=True)
    %% =========================================================

    % for i2 = 1:length(st_cp)
    %     x = double(st_cp(i2).d(:));
    %     st_cp(i2).d = x - mean(x);
    % end
    for i2 = 1:length(st_cp)
    x = double(st_cp(i2).d(:));

    mean_x = mean(x);

    %fprintf('Component %d, mean(x) = %.17g\n', i2, mean_x);

    st_cp(i2).d = x - mean_x;
    end
%==========================================================================

    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage01_demean.csv'], 'stage01_demean');
    end

    for i2 = 1:length(st_cp)
        x = double(st_cp(i2).d(:));
        st_cp(i2).d = detrend(x, 'linear');
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage02_detrend.csv'], 'stage02_detrend');
    end

    for i2 = 1:length(st_cp)
        x = double(st_cp(i2).d(:));
        [x_tapered, ~] = obspy_trace_taper_cosine_exact(x, 0.05);
        st_cp(i2).d = x_tapered(:);
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage03_taper.csv'], 'stage03_taper');
    end

    for i2 = 1:length(st_cp)
        x = double(st_cp(i2).d(:));
        st_cp(i2).d = obspy_filter_exact(x, Fs, fmin, fmax, 4, true, true);
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage04_filter.csv'], 'stage04_filter');
    end

    %% =========================================================
    % Resample to Python design frequency 100 Hz
    %% =========================================================
    Freq_pb = 100;
    if Fs ~= Freq_pb
        for i2 = 1:length(st_cp)
            st_cp(i2).d = resample(double(st_cp(i2).d(:)), Freq_pb, Fs);
            st_cp(i2).SampleRate = Freq_pb;
        end
        Fs = Freq_pb;
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage05_resample.csv'], 'stage05_resample');
    end

    %% =========================================================
    % Second offset elimination
    %% =========================================================
    for i2 = 1:length(st_cp)
        x = double(st_cp(i2).d(:));
        st_cp(i2).d = x - mean(x);
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage06_second_demean.csv'], 'stage06_second_demean');
    end

    %% =========================================================
    % Instrument correction
    % Python equivalent:
    %   attach_paz(st_cp[i2], pzfile, todisp=False, torad=False)
    %   zeros = np.array(st_cp[i2].stats.paz.zeros)
    %   zeros = np.delete(zeros, np.argwhere(zeros==0)[0:2])
    %   poles = np.array(st_cp[i2].stats.paz.poles)
    %   constant = st_cp[i2].stats.paz.gain
    %   sts2 = {'gain': constant, 'poles': poles, 'sensitivity': 1, 'zeros': zeros}
    %   st_cp[i2].simulate(paz_remove=sts2)
    %% =========================================================
    if pb_inst == true
        if isempty(pzfile)
            error('pb_inst is true, but pzfile is empty. Please provide a SAC Pole-Zero file path.');
        end
        if ~isfile(pzfile)
            error('PZ file not found: %s', string(pzfile));
        end

        paz = read_sac_pz_matlab(pzfile);

        % Python removes two zeros equal to 0+0i before simulate().
        zero_zero_idx = find(abs(paz.zeros) == 0);
        if numel(zero_zero_idx) >= 2
            paz.zeros(zero_zero_idx(1:2)) = [];
        else
            warning('Less than two zero-valued PAZ zeros found in pzfile. No two-zero removal was applied exactly.');
        end

        paz.sensitivity = 1.0;

        for i2 = 1:length(st_cp)
            x = double(st_cp(i2).d(:));
            st_cp(i2).d = obspy_simulate_paz_remove_matlab(x, Fs, paz);
            st_cp(i2).SampleRate = Fs;
        end

        if save_debug
            save_stage_csv(st_cp, Fs, [debug_prefix '_stage06b_instrument_correction.csv'], 'stage06b_instrument_correction');
        end
    end

    %% =========================================================
    % Second signal conditioning
    % Python:
    %    st_cp.detrend('demean')
    %    st_cp.detrend('linear')
    %    st_cp.taper(0.05, 'cosine', side='both')
    %    st_cp.filter('bandpass', ..., corners=4, zerophase=True)
    %% =========================================================

    for i2 = 1:length(st_cp)
        x = double(st_cp(i2).d(:));
        st_cp(i2).d = x - mean(x);
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage07_second_demean.csv'], 'stage07_second_demean');
    end

    for i2 = 1:length(st_cp)
        x = double(st_cp(i2).d(:));
        st_cp(i2).d = detrend(x, 'linear');
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage08_second_detrend.csv'], 'stage08_second_detrend');
    end

    for i2 = 1:length(st_cp)
        x = double(st_cp(i2).d(:));
        [x_tapered, ~] = obspy_trace_taper_cosine_exact(x, 0.05);
        st_cp(i2).d = x_tapered(:);
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage09_second_taper.csv'], 'stage09_second_taper');
    end

    for i2 = 1:length(st_cp)
        x = double(st_cp(i2).d(:));
        st_cp(i2).d = obspy_filter_exact(x, Fs, fmin, fmax, 4, true, false);
        st_cp(i2).SampleRate = Fs;
    end
    if save_debug
        save_stage_csv(st_cp, Fs, [debug_prefix '_stage10_second_filter.csv'], 'stage10_second_filter');
    end

    %% =========================================================
    % BAZ features
    %% =========================================================
    data_baz = [double(st_cp(1).d(:))'; ...
                double(st_cp(2).d(:))'; ...
                double(st_cp(3).d(:))'];

    % % C = cov(data_baz');
    % % [V, D] = eig(C);
    % % w = diag(D);
    % % w_sorted = sort(w);
    % % [~, idx_max] = max(w);
    % % 
    % % baz_features = [w_sorted(end), ...
    % %                 w_sorted(end) / (w_sorted(end-1) + w_sorted(end-2)), ...
    % %                 V(:, idx_max)'];



    %=====================版本2=======================（DET predition 结果为最优）
    % C = cov(data_baz');
    % [V, D] = eig(C);
    % 
    % % ===== Debug information =====
    % debug_info.data_baz = data_baz;
    % debug_info.cov_matrix = C;
    % debug_info.eigenvalues = diag(D);
    % debug_info.eigenvectors = V;
    % 
    % 
    % 
    % 
    % w = diag(D);
    % w_sorted = sort(w);
    % [~, idx_max] = max(w);
    % 
    % v_main = V(:, idx_max);
    % 
    % % Fix eigenvector sign to match Python-style deterministic direction
    % % Eigenvectors v and -v are equivalent, but XGBoost needs the same sign.
    % % if v_main(1) < 0
    % %     v_main = -v_main;
    % % end
    % 
    % baz_features = [w_sorted(end), ...
    %                 w_sorted(end) / (w_sorted(end-1) + w_sorted(end-2)), ...
    %                 v_main'];

    %======================版本3==================================


    % C = cov(data_baz');
    % 
    % [V, D] = eig(C);
    % w = diag(D);
    % 
    % % Sort like numpy.linalg.eigh:
    % % eigenvalues ascending, eigenvectors follow the same order
    % [w_sorted, order] = sort(w, 'ascend');
    % V_sorted = V(:, order);
    % 
    % % Largest eigenvalue eigenvector, same as eig_vecs[:, -1] in numpy
    % v_main = V_sorted(:, end);
    % 
    % % Reproduce numpy/LAPACK-like eigenvector sign convention:
    % % choose the element with largest absolute value;
    % % make that dominant element positive.
    % [~, idx_dom] = max(abs(v_main));
    % 
    % if v_main(idx_dom) < 0
    %     v_main = -v_main;
    % end
    % 
    % baz_features = [w_sorted(end), ...
    %                 w_sorted(end) / (w_sorted(end-1) + w_sorted(end-2)), ...
    %                 v_main'];



    %===================================版本4（调用NumPy)=================
    %% =========================================================
    % BAZ covariance and eigendecomposition using NumPy
    %
    % Exact Python counterpart:
    % data_baz = np.array([E, N, Z])
    % cov = np.cov(data_baz)
    % w, v = np.linalg.eig(cov)
    %% =========================================================
    
    np_local = py.importlib.import_module('numpy');
    
    % data_baz is 3 x N:
    % row 1 = E, row 2 = N, row 3 = Z
    data_baz_py = np_local.array(data_baz);
    
    % Exact counterpart of:
    % C = np.cov(data_baz)
    %
    % NumPy treats each row as one variable by default.
    C_py = np_local.cov(data_baz_py);
    
    % Exact counterpart of:
    % w, v = np.linalg.eig(C)
    eig_result = np_local.linalg.eig(C_py);
    
    w_py = eig_result{1};
    V_py = eig_result{2};
    
    %% Convert NumPy covariance matrix to MATLAB
    
    C_vec = double(py.array.array( ...
        'd', py.numpy.nditer(C_py)));
    
    % np.nditer traverses the NumPy matrix row by row.
    % MATLAB reshape fills matrices column by column,
    % so transpose after reshape.
    C = reshape(C_vec, 3, 3).';
    
    %% Convert NumPy eigenvalues to MATLAB
    
    w = double(py.array.array( ...
        'd', py.numpy.nditer(w_py)));
    
    w = real(w(:));
    
    %% Convert NumPy eigenvectors to MATLAB
    
    V_vec = double(py.array.array( ...
        'd', py.numpy.nditer(V_py)));
    
    V = reshape(V_vec, 3, 3).';
    V = real(V);
    
    %% Save debug information
    
    debug_info.data_baz = data_baz;
    debug_info.cov_matrix = C;
    debug_info.eigenvalues = w;
    debug_info.eigenvectors = V;
    
    %% Construct BAZ features
    
    % Exact counterpart of:
    % w_sorted = np.sort(w)
    w_sorted = sort(w);
    
    % Exact counterpart of:
    % np.argmax(w)
    [~, idx_max] = max(w);
    
    % Exact counterpart of:
    % v[:, np.argmax(w)]
    v_main = V(:, idx_max);
    
    % Do not apply any additional sign correction here.
    baz_features = [ ...
        w_sorted(end), ...
        w_sorted(end) / ...
            (w_sorted(end-1) + w_sorted(end-2)), ...
        v_main.' ...
    ];

    %=========================================================================

    FV = [FV, baz_features];
    debug_info.baz_features = baz_features;

    if save_debug
        write_vector_csv([debug_prefix '_baz_features.csv'], baz_features, 'BAZ');
    end

    %% =========================================================
    % Component-wise features
    % Per component: 45 features
    %   1-7   temporal
    %   8-17  envelope
    %   18-32 spectrum
    %   33-45 MFCC
    %% =========================================================
    component_rows = [];

    for i2 = 1:length(st_cp)

        Fs_i = double(st_cp(i2).SampleRate);
        x = double(st_cp(i2).d(:));
        N = length(x);
        time = (0:N-1)' / Fs_i;

        temp_features = time_features_7_exact(x, time);

        analytic_signal_t = hilbert(x);
        amp_env_t = abs(analytic_signal_t);
        env_features = envelope_features_10_exact(x, amp_env_t, Fs_i);

        N_fft = 1024;
        per_overlap = 75;
        if length(x) < N_fft
            N_fft = length(x);
        end
        N_overlap = floor(N_fft * per_overlap / 100);
        [Pxx, f] = pwelch_python_style(x, Fs_i, N_fft, N_overlap);
        spec_features = spectrum_features_15_exact(Pxx, f);

        mfcc_feat = python_speech_features_mfcc_1frame( ...
            x, Fs_i, N, 13, 26, 1.0, Fs_i/2, 0.97, 22);

        comp_features = [
            temp_features(:);
            env_features(:);
            spec_features(:);
            mfcc_feat(:)
        ];

        FV = [FV, comp_features(:)'];

        if save_debug
            feature_index = (1:length(comp_features))';
            component_rows = [component_rows; ... %#ok<AGROW>
                [repmat(i2-1, length(comp_features), 1), feature_index, comp_features(:)]];
        end
    end

    FV = real(FV);
    debug_info.FV = FV;

    if save_debug
        component_table = array2table(component_rows, ...
            'VariableNames', {'component_index','feature_index_in_component','value'});
        writetable(component_table, [debug_prefix '_component_features.csv']);
        write_vector_csv([debug_prefix '_FV.csv'], FV, 'FV');
    end
end

%% ============================================================
% Debug CSV helper functions
%% ============================================================
function save_stage_csv(st_cp, Fs, filename, stage_name)
    rows = [];
    comp_names = {'E','N','Z'};
    for c = 1:length(st_cp)
        x = double(st_cp(c).d(:));
        n = length(x);
        sample_index = (0:n-1)';
        time_s = sample_index / Fs;
        rows = [rows; ... %#ok<AGROW>
            table(repmat({stage_name}, n, 1), ...
                  repmat(c-1, n, 1), ...
                  repmat(comp_names(c), n, 1), ...
                  repmat(Fs, n, 1), ...
                  sample_index, ...
                  time_s, ...
                  x, ...
                  'VariableNames', {'stage','component_index','component','sampling_rate','sample_index','time_s','amplitude'})];
    end
    writetable(rows, filename);
end

function write_vector_csv(filename, vec, prefix)
    vec = real(vec(:)');
    T = array2table(vec);
    names = cell(1, numel(vec));
    for k = 1:numel(vec)
        names{k} = sprintf('%s_%03d', prefix, k);
    end
    T.Properties.VariableNames = names;
    writetable(T, filename);
end

%% ============================================================
% SAC Pole-Zero parser and PAZ deconvolution
%% ============================================================
function paz = read_sac_pz_matlab(pzfile)

    txt = fileread(pzfile);
    lines = regexp(txt, '\r\n|\n|\r', 'split');

    zeros_v = complex([]);
    poles_v = complex([]);
    constant = [];

    mode = '';
    expected = 0;
    count = 0;

    for k = 1:numel(lines)
        line = strtrim(lines{k});
        if isempty(line) || startsWith(line, '*') || startsWith(line, '#')
            continue;
        end

        parts = regexp(line, '\s+', 'split');
        key = upper(parts{1});

        if strcmp(key, 'ZEROS')
            mode = 'ZEROS';
            expected = str2double(parts{2});
            count = 0;
            continue;
        elseif strcmp(key, 'POLES')
            mode = 'POLES';
            expected = str2double(parts{2});
            count = 0;
            continue;
        elseif strcmp(key, 'CONSTANT')
            constant = str2double(parts{2});
            mode = '';
            continue;
        end

        if strcmp(mode, 'ZEROS') && count < expected
            if numel(parts) >= 2
                re = str2double(parts{1});
                im = str2double(parts{2});
            else
                re = 0; im = 0;
            end
            zeros_v(end+1,1) = complex(re, im); %#ok<AGROW>
            count = count + 1;
        elseif strcmp(mode, 'POLES') && count < expected
            re = str2double(parts{1});
            im = str2double(parts{2});
            poles_v(end+1,1) = complex(re, im); %#ok<AGROW>
            count = count + 1;
        end
    end

    if isempty(constant) || isnan(constant)
        error('Could not read CONSTANT from PZ file: %s', string(pzfile));
    end

    paz = struct();
    paz.zeros = zeros_v(:);
    paz.poles = poles_v(:);
    paz.gain = constant;
    paz.sensitivity = 1.0;
end

function y = obspy_simulate_paz_remove_matlab(x, Fs, paz)
% MATLAB translation of ObsPy invsim.simulate_seismometer() for the exact
% call used in Python:
%   Trace.simulate(paz_remove=sts2)
%
% ObsPy defaults used here:
%   water_level=600, zero_mean=true, taper=true, taper_fraction=0.05,
%   pre_filt=None, nfft_pow2=false, pitsasim=true, sacsim=false,
%   shsim=false, remove_sensitivity=true.
%
% The important differences from the earlier approximate version are:
%   1) nfft is at least 2*ndat, following ObsPy _npts2nfft().
%   2) the operation uses an rFFT-style one-sided spectrum.
%   3) the PAZ response is inverted with ObsPy's invert_spectrum logic.
%   4) after irFFT, ObsPy applies simple_detrend() when pitsasim=true.

    x = double(x(:));
    ndat = length(x);
    delta = 1.0 / Fs;

    % ObsPy: data = data.astype(np.float64)
    data = x;

    % ObsPy default: zero_mean=True
    data = data - mean(data);

    % ObsPy default: taper=True, sacsim=False -> cosine_taper(ndat, 0.05)
    [~, taper_win] = obspy_trace_taper_cosine_exact(data, 0.05);
    data = data .* taper_win(:);

    % ObsPy default: nfft_pow2=False -> _npts2nfft(ndat)
    nfft = obspy_npts2nfft_matlab(ndat);

    % ObsPy: data = np.fft.rfft(data, n=nfft)
    data_spec_full = fft(data, nfft);
    n_rfft = floor(nfft / 2) + 1;
    data_spec = data_spec_full(1:n_rfft);

    % ObsPy: paz_to_freq_resp(..., freq=True)
    [freq_response, ~] = paz_to_freq_resp_matlab( ...
        paz.poles, paz.zeros, paz.gain, delta, nfft);

    % ObsPy: invert_spectrum(freq_response, water_level)
    freq_response = invert_spectrum_matlab(freq_response, 600.0);

    % ObsPy: data *= freq_response
    data_spec = data_spec .* freq_response;

    % ObsPy: data[-1] = abs(data[-1]) + 0.0j
    data_spec(end) = abs(data_spec(end)) + 0.0i;

    % ObsPy: data = np.fft.irfft(data)[0:ndat]
    if rem(nfft, 2) == 0
        full_spec = [data_spec; conj(data_spec(end-1:-1:2))];
    else
        full_spec = [data_spec; conj(data_spec(end:-1:2))];
    end
    data_time = real(ifft(full_spec, nfft));
    data_time = data_time(1:ndat);

    % ObsPy default: pitsasim=True -> simple_detrend(data)
    data_time = obspy_simple_detrend_matlab(data_time);

    % ObsPy: if paz_remove and remove_sensitivity and not seedresp:
    %            data /= paz_remove['sensitivity']
    if isfield(paz, 'sensitivity') && ~isempty(paz.sensitivity)
        data_time = data_time ./ paz.sensitivity;
    end

    y = data_time(:);
end

function nfft = obspy_npts2nfft_matlab(npts)
% Translation of obspy.signal.util._npts2nfft(npts) for the current window
% sizes. For nfft <= 5000, ObsPy does not run the smart factorisation branch.
    if bitand(npts, 1) ~= 0
        nfft = 2 * (npts + 1);
    else
        nfft = 2 * npts;
    end

    % This project's 10 s, 100 Hz debug windows give ndat=1001 and nfft=2004,
    % so the branch below is normally not used. It is included for safety.
    if nfft > 5000 && ~obspy_good_factorization_matlab(nfft)
        found = false;
        for i_try = 1:10
            trial = nfft + 2 * i_try;
            if obspy_good_factorization_matlab(trial)
                nfft = trial;
                found = true;
                break;
            end
        end
        if ~found
            nfft = 2 ^ nextpow2(nfft);
        end
    end
end

function tf = obspy_good_factorization_matlab(x)
    pf = factor(x);
    tf = max(pf) < 500;
end

function [h, f] = paz_to_freq_resp_matlab(poles, zeros_v, scale_fac, t_samp, nfft)
% Translation of ObsPy paz_to_freq_resp(). ObsPy uses scipy.signal.freqs()
% after zpk2tf; evaluating the ZPK product at jw gives the same analog
% response on the same frequency grid.
    n = floor(nfft / 2);
    fy = 1 / (t_samp * 2.0);
    f = linspace(0, fy, n + 1).';
    s = 1i * 2 * pi * f;

    h = scale_fac * ones(n + 1, 1);
    for iz = 1:numel(zeros_v)
        h = h .* (s - zeros_v(iz));
    end
    for ip = 1:numel(poles)
        h = h ./ (s - poles(ip));
    end
end

function spec_out = invert_spectrum_matlab(spec, wlev)
% Translation of ObsPy invert_spectrum(spec, wlev).
% It first raises amplitudes below the water level, preserving phase, then
% inverts non-zero spectral values.
    spec_out = spec(:);
    swamp = max(abs(spec_out)) * 10.0 ^ (-wlev / 20.0);

    sqrt_len = abs(spec_out);
    idx = (sqrt_len < swamp) & (sqrt_len > 0.0);
    spec_out(idx) = spec_out(idx) .* (swamp ./ sqrt_len(idx));

    sqrt_len = abs(spec_out);
    inn = sqrt_len > 0.0;
    spec_out(inn) = 1.0 ./ spec_out(inn);
    spec_out(sqrt_len == 0.0) = complex(0.0, 0.0);
end

function data = obspy_simple_detrend_matlab(data)
% Translation of obspy.signal.detrend.simple(): subtract a line through the
% first and last sample.
    data = double(data(:));
    ndat = length(data);
    if ndat <= 1
        return;
    end
    x1 = data(1);
    x2 = data(end);
    data = data - (x1 + (0:ndat-1).' * (x2 - x1) / double(ndat - 1));
end

%% ============================================================
% Exact ObsPy cosine taper translation
%% ============================================================
function [x_tapered, taper] = obspy_trace_taper_cosine_exact(x, max_percentage)

    original_size = size(x);
    x_col = double(x(:));
    npts = length(x_col);

    if max_percentage < 0 || max_percentage > 0.5
        error('max_percentage must be between 0 and 0.5.');
    end

    wlen = floor(max_percentage * npts);

    if 2 * wlen == npts
        taper_sides = obspy_cosine_taper_window_only(2 * wlen, 1.0);
    else
        taper_sides = obspy_cosine_taper_window_only(2 * wlen + 1, 1.0);
    end

    taper = [
        taper_sides(1:wlen);
        ones(npts - 2 * wlen, 1);
        taper_sides(end-wlen+1:end)
    ];

    x_tapered_col = x_col .* taper;
    x_tapered = reshape(x_tapered_col, original_size);
end

function taper = obspy_cosine_taper_window_only(npts, p)

    if p < 0 || p > 1
        error('Decimal taper percentage must be between 0 and 1.');
    end

    if p == 0.0 || p == 1.0
        frac = floor(npts * p / 2.0);
    else
        frac = floor(npts * p / 2.0 + 0.5);
    end

    idx1_py = 0;
    idx2_py = frac - 1;
    idx3_py = npts - frac;
    idx4_py = npts - 1;

    if idx1_py == idx2_py
        idx2_py = idx2_py + 1;
    end

    if idx3_py == idx4_py
        idx3_py = idx3_py - 1;
    end

    taper = zeros(npts, 1);

    idx1 = idx1_py + 1;
    idx2 = idx2_py + 1;
    idx3 = idx3_py + 1;
    idx4 = idx4_py + 1;

    k_left = (idx1_py:idx2_py)';
    taper(idx1:idx2) = 0.5 * ...
        (1.0 - cos(pi * (k_left - double(idx1_py)) / double(idx2_py - idx1_py)));

    taper(idx2+1:idx3-1) = 1.0;

    k_right = (idx3_py:idx4_py)';
    taper(idx3:idx4) = 0.5 * ...
        (1.0 + cos(pi * (double(idx3_py) - k_right) / double(idx4_py - idx3_py)));
end

%% ============================================================
% Exact ObsPy-style Butterworth filter
%% ============================================================
function y = obspy_filter_exact(x, Fs, fmin, fmax, corners, zerophase, allow_highpass_if_fmax_above_nyq)

    x = double(x(:));
    nyq = Fs / 2.0;

    if allow_highpass_if_fmax_above_nyq && nyq < fmax
        Wn = fmin / nyq;
        [z, p, k] = butter(corners, Wn, 'high');
    else
        Wn = [fmin / nyq, fmax / nyq];
        [z, p, k] = butter(corners, Wn, 'bandpass');
    end

    [sos, g] = zp2sos(z, p, k);
    sos(1, 1:3) = sos(1, 1:3) * g;

    y = sosfilt(sos, x);

    if zerophase
        y = flipud(y);
        y = sosfilt(sos, y);
        y = flipud(y);
    end
end

%% ============================================================
% Feature functions: temporal, envelope, spectrum
%% ============================================================
function feat = time_features_7_exact(x, time)

    x = double(x(:));
    time = double(time(:));

    Ene_t = x .^ 2;
    Et = sum(Ene_t);

    maxEt = max(Ene_t);

    [~, idx_max] = max(Ene_t);
    argmaxEt = time(idx_max);

    centroid_t = sum(time .* Ene_t) / Et;

    BW_t = sqrt(sum(((time - centroid_t) .^ 2) .* Ene_t) / Et);

    skew_raw = sum(((time - centroid_t) .^ 3) .* Ene_t) / ...
        (Et * BW_t^3);

    if skew_raw >= 0
        skewness_t = sqrt(skew_raw);
    else
        skewness_t = -sqrt(-skew_raw);
    end

    kurtosis_t = sqrt(sum(((time - centroid_t) .^ 4) .* Ene_t) / ...
        (Et * BW_t^4));

    energy_sum = Et;

    feat = [
        maxEt;
        argmaxEt;
        centroid_t;
        BW_t;
        skewness_t;
        kurtosis_t;
        energy_sum
    ];
end

function feat = envelope_features_10_exact(x, env, Fs)

    x = double(x(:));
    env = double(env(:));

    TCR_t_val = TCR_t_exact(env, 0.8, Fs);
    RMM_val = max(env) / mean(env);
    mean_env = mean(env);

    std_env = std(env, 1);
    skew_env = skewness(env, 1);
    kurt_env = kurtosis(env, 1);

    mTCR_val = mTCR_t_exact(env, 0.8);
    shannon_val = shannon_ent_numpy_exact(env, 200);
    renyi_val = renyi_ent_numpy_exact(env, 2, 200);
    ZCR_val = ZCR_t_exact(x, Fs);

    feat = [
        TCR_t_val;
        RMM_val;
        mean_env;
        std_env;
        skew_env;
        kurt_env;
        mTCR_val;
        shannon_val;
        renyi_val;
        ZCR_val
    ];
end

function feat = spectrum_features_15_exact(PSD, f)

    PSD = double(PSD(:));
    f = double(f(:));

    maxPSD = max(PSD);
    argmaxPSD = f(find(PSD == maxPSD, 1, 'first'));

    Ef = sum(PSD);

    centroid = sum(f .* PSD) / Ef;

    BW = sqrt(sum(((f - centroid) .^ 2) .* PSD) / Ef);

    skew_raw = sum(((f - centroid) .^ 3) .* PSD) / ...
        (Ef * BW^3);

    if skew_raw >= 0
        skewness_f = sqrt(skew_raw);
    else
        skewness_f = -sqrt(-skew_raw);
    end

    kurtosis_f = sqrt(sum(((f - centroid) .^ 4) .* PSD) / ...
        (Ef * BW^4));

    meanPSD = mean(PSD);
    stdPSD = std(PSD, 1);
    skewPSD = skewness(PSD, 1);
    kurtPSD = kurtosis(PSD, 1);

    shannonPSD = shannon_ent_numpy_exact(PSD, 50);
    renyiPSD = renyi_ent_numpy_exact(PSD, 2, 50);

    RMM_PSD = max(PSD) / mean(PSD);

    TCR_f_val = TCR_f_exact(PSD, 0.4);
    mTCR_val = mTCR_t_exact(PSD, 0.4);

    feat = [
        maxPSD;
        argmaxPSD;
        centroid;
        BW;
        skewness_f;
        kurtosis_f;
        meanPSD;
        stdPSD;
        skewPSD;
        kurtPSD;
        shannonPSD;
        renyiPSD;
        RMM_PSD;
        TCR_f_val;
        mTCR_val
    ];
end

%% ============================================================
% Python-style Welch PSD
%% ============================================================
function [Pxx, f] = pwelch_python_style(x, Fs, N_fft, N_overlap)

    x = double(x(:));

    window = hann(N_fft, 'periodic');
    step = N_fft - N_overlap;
    n = length(x);

    starts = 1:step:(n - N_fft + 1);
    nseg = length(starts);

    U = sum(window .^ 2);

    n_freq = floor(N_fft / 2) + 1;
    Pxx_acc = zeros(n_freq, 1);

    for i = 1:nseg
        seg = x(starts(i):starts(i)+N_fft-1);

        segw = seg .* window;
        X = fft(segw, N_fft);

        P2 = (abs(X) .^ 2) / (Fs * U);
        P1 = P2(1:n_freq);

        if rem(N_fft, 2) == 0
            P1(2:end-1) = 2 * P1(2:end-1);
        else
            P1(2:end) = 2 * P1(2:end);
        end

        Pxx_acc = Pxx_acc + P1;
    end

    Pxx = Pxx_acc / nseg;
    f = (0:n_freq-1)' * Fs / N_fft;
end

%% ============================================================
% Python_speech_features MFCC one-frame equivalent
%% ============================================================
function ceps = python_speech_features_mfcc_1frame(x, Fs, nfft, numcep, nfilt, lowfreq, highfreq, preemph, ceplifter)

    x = double(x(:));

    x_pre = [x(1); x(2:end) - preemph * x(1:end-1)];

    frame = x_pre(:).';

    mag = abs(fft(frame, nfft));

    pow = (1.0 / nfft) * (mag .^ 2);
    pow = pow(1:floor(nfft/2)+1);

    fb = get_filterbanks_python_style(nfilt, nfft, Fs, lowfreq, highfreq);

    feat = pow * fb.';
    feat(feat == 0) = eps;

    log_feat = log(feat);

    ceps = dct(log_feat);
    ceps = ceps(1:numcep);

    ceps = lifter_python_style(ceps(:), ceplifter);
end

function fb = get_filterbanks_python_style(nfilt, nfft, samplerate, lowfreq, highfreq)

    lowmel = hz2mel_python(lowfreq);
    highmel = hz2mel_python(highfreq);

    melpoints = linspace(lowmel, highmel, nfilt + 2);
    bin_hz = mel2hz_python(melpoints);

    bin = floor((nfft + 1) * bin_hz / samplerate);

    n_freq = floor(nfft / 2) + 1;
    fb = zeros(nfilt, n_freq);

    for j = 1:nfilt
        left = bin(j);
        center = bin(j+1);
        right = bin(j+2);

        for i = left:center
            matlab_idx = i + 1;
            if matlab_idx >= 1 && matlab_idx <= n_freq
                fb(j, matlab_idx) = (i - left) / (center - left);
            end
        end

        for i = center:right
            matlab_idx = i + 1;
            if matlab_idx >= 1 && matlab_idx <= n_freq
                fb(j, matlab_idx) = (right - i) / (right - center);
            end
        end
    end
end

%% ============================================================
% Scalar helper functions matching verified Python behavior
%% ============================================================
function val = TCR_t_exact(Xn, thres, Fs)
    duration = length(Xn) / Fs;
    Xn = Xn / max(abs(Xn));
    Xn = Xn - thres;
    val = sum((Xn(1:end-1) .* Xn(2:end)) < 0) / duration;
end

function val = ZCR_t_exact(Xn, Fs)
    duration = length(Xn) / Fs;
    val = sum((Xn(1:end-1) .* Xn(2:end)) < 0) / duration;
end

function val = TCR_f_exact(PSD, thres)
    PSD = double(PSD(:));
    PSD = PSD / max(abs(PSD));
    PSD = PSD - thres;
    val = sum((PSD(1:end-1) .* PSD(2:end)) < 0);
end

function val = mTCR_t_exact(Xn, thres)
    Xn = double(Xn(:));
    Xn = Xn / max(abs(Xn));

    TH = Xn;
    mask = (TH < thres) | (TH > 1);
    TH(mask) = -127;

    idx = find(TH ~= -127);
    val = length(idx) / length(Xn);
end

function val = shannon_ent_numpy_exact(Xn, Bins)

    Xn = double(Xn(:));

    edges = linspace(min(Xn), max(Xn), Bins + 1);
    counts = histcounts(Xn, edges);

    prob = counts / length(Xn);
    prob = prob(prob ~= 0);

    val = sum(-prob .* log2(prob));
end

function val = renyi_ent_numpy_exact(Xn, alpha, Bins)

    Xn = double(Xn(:));

    edges = linspace(min(Xn), max(Xn), Bins + 1);
    counts = histcounts(Xn, edges);

    prob = counts / length(Xn);
    prob = prob(prob ~= 0);

    val = log2(sum(prob .^ alpha)) / (1 - alpha);
end

function mel = hz2mel_python(hz)
    mel = 2595 * log10(1 + hz / 700);
end

function hz = mel2hz_python(mel)
    hz = 700 * (10 .^ (mel / 2595) - 1);
end

function y = lifter_python_style(cepstra, L)
    if L > 0
        n = (0:length(cepstra)-1)';
        lift = 1 + (L / 2) * sin(pi * n / L);
        y = lift .* cepstra;
    else
        y = cepstra;
    end
end

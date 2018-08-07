function [ann, anntype] = wrapper (tm, ecg, outfile, fs)
    tm = cell2mat(tm)';
    ecg = cell2mat(ecg)';
    wrsamp(tm,ecg,outfile, fs);
    ecgpuwave(outfile,'test');
    [ann, anntype] = rdann(outfile,'test');
end
% test
function [onsets, feats, BeatQ, R] = wabp_wrap (py_ABP)

    Fwf = 125;
    Fs  = 240;
    
    ABP = cell2mat(py_ABP)';
    ABP = resample(ABP,Fwf,Fs);
    onsets = wabp(ABP);
    if length (onsets) > 1
        feats  = abpfeature(ABP, onsets);
        [BeatQ, R] = jSQI(feats, onsets, ABP);
    else
        feats = [];
        BeatQ = 0;
        R = [];
    

end
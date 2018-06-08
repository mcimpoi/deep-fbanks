function mc_fix_nn_compatibility(model_path)
%MC_FIX_NN_COMPATIBILITY Summary of this function goes here
%   Detailed explanation goes here
    if ~exist(model_path)
        return
    end
    load(model_path);
    for ii = 1 : length(layers)
        if strcmp(layers{ii}.type, 'lrn')
            layers{ii}.type = 'normalize';
        end
    end
    normalization = meta.normalization;
    save(model_path, 'layers', 'meta', 'normalization');
end


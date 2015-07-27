function encoder_save(encoder, filePath)
if isfield(encoder, 'net')
  encoder.net = vl_simplenn_move(encoder.net, 'cpu') ;
  encoder.net.useGpu = false ;
end
save(filePath, '-struct', 'encoder') ;

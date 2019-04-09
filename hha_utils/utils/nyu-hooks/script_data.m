nyufile = fullfile('/Users/Or Guz/Desktop/project/3DGNN_pytorch/datasets/data', 'nyu_depth_v2_labeled.mat');
outDir = fullfile('../nyu/data');
mkdir(fullfile(outDir, 'rawdepth'));
mkdir(fullfile(outDir, 'depth'));
mkdir(fullfile(outDir, 'images'));


dt = load(nyufile, 'rawDepths');
for i = 1:300, %1449 
  imwrite(uint16(cropIt(dt.rawDepths(:,:,i))*1000), fullfile(outDir, 'rawdepth', sprintf('img_%04d.png', i + 5000))); 
end

dt = load(nyufile, 'depths');
for i = 1:300, 
  imwrite(uint16(cropIt(dt.depths(:,:,i))*1000), fullfile(outDir, 'depth', sprintf('img_%04d.png', i + 5000))); 
end

dt = load(nyufile, 'images');
for i = 1:300, 
  imwrite(uint8(cropIt(dt.images(:,:,:,i))), fullfile(outDir, 'images', sprintf('img_%04d.png', i + 5000))); 
end

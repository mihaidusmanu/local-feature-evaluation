function features_from_mat(METHOD_NAME, num_images, image_names, image_paths, keypoint_paths, descriptor_paths)

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(maxNumCompThreads());
end

parfor i = 1:num_images
    fprintf('Computing features for %s [%d/%d]', ...
            image_names{i}, i, num_images);

    tic;
    
    data = load([image_paths{i} '.' METHOD_NAME], '-mat');
    keypoints = [data.keypoints(:, [1 2]) zeros(size(data.keypoints, 1), 2)];
    descriptors = data.descriptors;

    write_keypoints(keypoint_paths{i}, keypoints);

    write_descriptors(descriptor_paths{i}, descriptors);

    fprintf(' in %.3fs\n', toc);
end

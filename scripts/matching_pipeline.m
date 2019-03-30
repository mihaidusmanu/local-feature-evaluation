% Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>
function matching_pipeline(METHOD_NAME, DATASET_NAMES)

clc;

for i = 1:length(DATASET_NAMES)
    disp(DATASET_NAMES{i});

    %% Set the pipeline parameters.

    % TODO: Change this to where your dataset is stored. This directory should
    %       contain an "images" folder and a "database.db" file.
    DATASET_PATH = ['/local/dataset/local-feature-evaluation/' DATASET_NAMES{i}];

    % TODO: Change this to where VLFeat is located.
    %VLFEAT_PATH = '/home/mihai/sources/tools/vlfeat-0.9.21';

    % TODO: Change this to where the COLMAP build directory is located.
    %COLMAP_PATH = '/home/mihai/sources/tools/colmap/build';

    % Radius of local patches around each keypoint.
    %PATCH_RADIUS = 32;

    % Whether to run matching on GPU.
    MATCH_GPU = gpuDeviceCount() > 0;

    % Number of images to match in one block.
    MATCH_BLOCK_SIZE = 50;

   % Maximum distance ratio between first and second best matches.
    MATCH_MAX_DIST_RATIO = 0.8;

    % Minimum number of matches between two images.
    MIN_NUM_MATCHES = 15;

    %% Setup the pipeline environment.
    %run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));

    IMAGE_PATH = fullfile(DATASET_PATH, 'images');
    KEYPOINT_PATH = fullfile(DATASET_PATH, ['keypoints-' METHOD_NAME]);
    DESCRIPTOR_PATH = fullfile(DATASET_PATH, ['descriptors-' METHOD_NAME]);
    MATCH_PATH = fullfile(DATASET_PATH, ['matches-' METHOD_NAME]);
    INITIAL_DATABASE_PATH = fullfile(DATASET_PATH, 'database.db');
    DATABASE_PATH = fullfile(DATASET_PATH, [METHOD_NAME '.db']);
    
    copyfile(INITIAL_DATABASE_PATH, DATABASE_PATH);

    %% Create the output directories.

    if ~exist(KEYPOINT_PATH, 'dir')
        mkdir(KEYPOINT_PATH);
    end
    if ~exist(DESCRIPTOR_PATH, 'dir')
        mkdir(DESCRIPTOR_PATH);
    end
    if ~exist(MATCH_PATH, 'dir')
        mkdir(MATCH_PATH);
    end

    %% Extract the image names and paths.

    image_files_all = dir(IMAGE_PATH);
    image_files = [];
    for i = 3 : length(image_files_all)
        ext = lower(image_files_all(i).name(end - 2 : end));
        if ~ (strcmp(ext, 'png')  || strcmp(ext, 'jpg'))
            continue
        end
        image_files = [image_files image_files_all(i)];
    end
    num_images = length(image_files);
    image_names = cell(num_images, 1);
    image_paths = cell(num_images, 1);
    keypoint_paths = cell(num_images, 1);
    descriptor_paths = cell(num_images, 1);
    for i = 1:length(image_files)
        image_name = image_files(i).name;
        image_names{i} = image_name;
        image_paths{i} = fullfile(IMAGE_PATH, image_name);
        keypoint_paths{i} = fullfile(KEYPOINT_PATH, [image_name '.bin']);
        descriptor_paths{i} = fullfile(DESCRIPTOR_PATH, [image_name '.bin']);
    end

    %% Compute the keypoints and descriptors.
    
    features_from_mat(METHOD_NAME, num_images, image_names, image_paths, keypoint_paths, descriptor_paths)

    %% Match the descriptors.
    %
    %  NOTE: - You must exhaustively match Fountain, Herzjesu, South Building,
    %          Madrid Metropolis, Gendarmenmarkt, and Tower of London.
    %        - You must approximately match Alamo, Roman Forum, Cornell.

    if num_images < 2000
        exhaustive_matching
    else
    %    VOCAB_TREE_PATH = fullfile(DATASET_PATH, 'Oxford5k/vocab-tree.bin');
    %    approximate_matching_new
    end
end

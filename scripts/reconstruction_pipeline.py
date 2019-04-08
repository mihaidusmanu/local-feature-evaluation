# Import the features and matches into a COLMAP database.
#
# Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

from __future__ import print_function, division

import os
import sys
import glob
import argparse
import sqlite3
import subprocess
import multiprocessing

import numpy as np

from tqdm import tqdm


IS_PYTHON3 = sys.version_info[0] >= 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset, e.g., path/to/Fountain")
    parser.add_argument("--colmap_path", default="/home/mihai/sources/tools/colmap/build/src/exe",
                        help="Path to the COLMAP executable folder, e.g., "
                             "path/to/colmap/build/src/exe")
    parser.add_argument("--method_name", required=True, 
                        help="Name of the method")
    args = parser.parse_args()
    return args


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)


def read_matrix_file(fid, dtype):
    shape = np.fromfile(fid, count=2, dtype=np.int32)
    matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)

def import_matches(args):
    connection = sqlite3.connect(os.path.join(args.dataset_path, args.method_name + ".db"))
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master "
                   "WHERE type='table' AND name='inlier_matches';")
    try:
        inlier_matches_table_exists = bool(next(cursor)[0])
    except StopIteration:
        inlier_matches_table_exists = False

    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    if inlier_matches_table_exists:
        cursor.execute("DELETE FROM inlier_matches;")
    else:
        cursor.execute("DELETE FROM two_view_geometries;")
    connection.commit()

    images = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]

    for image_name, image_id in tqdm(images.items(), total = len(images.items())):
        #print("Importing features for", image_name)
        keypoint_path = os.path.join(args.dataset_path, "keypoints-" + args.method_name,
                                     image_name + ".bin")
        keypoints = read_matrix(keypoint_path, np.float32)
        descriptor_path = os.path.join(args.dataset_path, "descriptors-" + args.method_name,
                                     image_name + ".bin")
        descriptors = read_matrix(descriptor_path, np.float32)
        assert keypoints.shape[1] == 4
        assert keypoints.shape[0] == descriptors.shape[0]
        if IS_PYTHON3:
            keypoints_str = keypoints.tostring()
        else:
            keypoints_str = np.getbuffer(keypoints)
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) "
                       "VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1],
                        keypoints_str))
        connection.commit()

    image_pairs = []
    image_pair_ids = set()
    pairs_file = open(os.path.join(args.dataset_path, "matches-" + args.method_name + "/pairs.txt"), "r")
    matches_file = open(os.path.join(args.dataset_path, "matches-" + args.method_name + "/matches.bin"), "rb")
    pairs = pairs_file.readlines()
    for pair in tqdm(pairs, total = len(pairs)):
        image_name1, image_name2 = pair.strip('\n').split(' ')
        image_pairs.append((image_name1, image_name2))
        #print("Importing matches for", image_name1, "---", image_name2)
        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        if image_pair_id in image_pair_ids:
            continue
        image_pair_ids.add(image_pair_id)
        matches = read_matrix_file(matches_file, np.uint32)
        assert matches.shape[1] == 2
        if IS_PYTHON3:
            matches_str = matches.tostring()
        else:
            matches_str = np.getbuffer(matches)
        cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) "
                       "VALUES(?, ?, ?, ?);",
                       (image_pair_id, matches.shape[0], matches.shape[1],
                        matches_str))
        connection.commit()
    pairs_file.close()
    matches_file.close()

    with open(os.path.join(args.dataset_path, "image-pairs.txt"), "w") as fid:
        for image_name1, image_name2 in image_pairs:
            fid.write("{} {}\n".format(image_name1, image_name2))

    cursor.close()
    connection.close()
    
    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                     "matches_importer",
                     "--database_path",
                     os.path.join(args.dataset_path, args.method_name + ".db"),
                     "--match_list_path",
                     os.path.join(args.dataset_path, "image-pairs.txt"),
                     "--match_type", "pairs"])
    
    connection = sqlite3.connect(os.path.join(args.dataset_path, args.method_name + ".db"))
    cursor = connection.cursor()

    cursor.execute("SELECT count(*) FROM images;")
    num_images = next(cursor)[0]

    cursor.execute("SELECT count(*) FROM two_view_geometries WHERE rows > 0;")
    num_inlier_pairs = next(cursor)[0]

    cursor.execute("SELECT sum(rows) FROM two_view_geometries WHERE rows > 0;")
    num_inlier_matches = next(cursor)[0]

    cursor.close()
    connection.close()

    return dict(num_images=num_images,
                num_inlier_pairs=num_inlier_pairs,
                num_inlier_matches=num_inlier_matches)


def reconstruct(args):
    database_path = os.path.join(args.dataset_path, args.method_name + ".db")
    image_path = os.path.join(args.dataset_path, "images")
    sparse_path = os.path.join(args.dataset_path, "sparse-" + args.method_name)
    dense_path = os.path.join(args.dataset_path, "dense-" + args.method_name)

    if not os.path.exists(sparse_path):
        os.makedirs(sparse_path)
    if not os.path.exists(dense_path):
        os.makedirs(dense_path)
    
    # Run the sparse reconstruction.

    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                     "mapper",
                     "--database_path", database_path,
                     "--image_path", image_path,
                     "--export_path", sparse_path,
                     "--Mapper.num_threads",
                     str(min(multiprocessing.cpu_count(), 16))])
    
    # Find the largest reconstructed sparse model.

    models = os.listdir(sparse_path)
    if len(models) == 0:
        print("Warning: Could not reconstruct any model")
        return

    largest_model = None
    largest_model_num_images = 0
    for model in models:
        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                         "model_converter",
                         "--input_path", os.path.join(sparse_path, model),
                         "--output_path", os.path.join(sparse_path, model),
                         "--output_type", "TXT"])
        with open(os.path.join(sparse_path, model, "cameras.txt"), "r") as fid:
            for line in fid:
                if line.startswith("# Number of cameras"):
                    num_images = int(line.split()[-1])
                    if num_images > largest_model_num_images:
                        largest_model = model
                        largest_model_num_images = num_images
                    break

    assert largest_model_num_images > 0

    # Run the dense reconstruction.

    largest_model_path = os.path.join(sparse_path, largest_model)
    workspace_path = os.path.join(dense_path, largest_model)
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                     "image_undistorter",
                     "--image_path", image_path,
                     "--input_path", largest_model_path,
                     "--output_path", workspace_path,
                     "--max_image_size", "1200"])

    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                     "patch_match_stereo",
                     "--workspace_path", workspace_path,
                     "--PatchMatchStereo.geom_consistency", "false"])

    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                     "stereo_fusion",
                     "--workspace_path", workspace_path,
                     "--input_type", "photometric",
                     "--output_path", os.path.join(workspace_path, "fused.ply"),
                     "--StereoFusion.min_num_pixels", "5"])

    stats = subprocess.check_output(
        [os.path.join(args.colmap_path, "colmap"), "model_analyzer",
         "--path", largest_model_path])

    stats = stats.decode().split("\n")
    for stat in stats:
        if stat.startswith("Registered images"):
            num_reg_images = int(stat.split()[-1])
        elif stat.startswith("Points"):
            num_sparse_points = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            num_observations = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            mean_track_length = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            num_observations_per_image = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            mean_reproj_error = float(stat.split()[-1][:-2])

    with open(os.path.join(workspace_path, "fused.ply"), "rb") as fid:
        line = fid.readline().decode()
        while line:
            if line.startswith("element vertex"):
                num_dense_points = int(line.split()[-1])
                break
            line = fid.readline().decode()

    return dict(num_reg_images=num_reg_images,
                num_sparse_points=num_sparse_points,
                num_observations=num_observations,
                mean_track_length=mean_track_length,
                num_observations_per_image=num_observations_per_image,
                mean_reproj_error=mean_reproj_error,
                num_dense_points=num_dense_points)


def main():
    args = parse_args()

    matching_stats = import_matches(args)
    reconstruction_stats = reconstruct(args)

    print()
    print(78 * "=")
    print("Raw statistics")
    print(78 * "=")
    print(matching_stats)
    print(reconstruction_stats)

    print()
    print(78 * "=")
    print("Formatted statistics")
    print(78 * "=")
    print("| " + " | ".join(
            map(str, [os.path.basename(args.dataset_path),
                      args.method_name,
                      matching_stats["num_images"],
                      reconstruction_stats["num_reg_images"],
                      reconstruction_stats["num_sparse_points"],
                      reconstruction_stats["num_observations"],
                      reconstruction_stats["mean_track_length"],
                      reconstruction_stats["num_observations_per_image"],
                      reconstruction_stats["mean_reproj_error"],
                      reconstruction_stats["num_dense_points"],
                      "",
                      "",
                      "",
                      "",
                      matching_stats["num_inlier_pairs"],
                      matching_stats["num_inlier_matches"]])) + " |")


if __name__ == "__main__":
    main()

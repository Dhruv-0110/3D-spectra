
import argparse
from pathlib import Path

import open3d as o3d


def reduce_point_cloud_noise(
    input_file,
    output_file,
    voxel_size=0.02,
    nb_neighbors=20,
    std_ratio=2.0,
):
    """
    Reduce point-cloud size and remove statistical outliers.

    Args:
        input_file: Path to the input .ply point cloud.
        output_file: Path for the cleaned .ply output.
        voxel_size: Size of each voxel, in the point cloud's units.
        nb_neighbors: Number of neighbours used for SOR.
        std_ratio: Outlier distance threshold multiplier.
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input point cloud not found: {input_path}"
        )

    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero.")

    if nb_neighbors < 2:
        raise ValueError("nb_neighbors must be at least 2.")

    if std_ratio <= 0:
        raise ValueError("std_ratio must be greater than zero.")

    print(f"Loading point cloud: {input_path}")

    pcd = o3d.io.read_point_cloud(str(input_path))

    original_count = len(pcd.points)

    if original_count == 0:
        raise ValueError(
            "The input file contains no readable points."
        )

    print(f"Original points: {original_count:,}")

    # Step 1: Reduce point density using voxel grid downsampling.
    print("Step 1: Applying voxel grid downsampling...")

    downsampled = pcd.voxel_down_sample(
        voxel_size=voxel_size
    )

    downsampled_count = len(downsampled.points)

    print(f"Points after downsampling: {downsampled_count:,}")

    # Step 2: Remove statistical outliers.
    print("Step 2: Applying statistical outlier removal...")

    if downsampled_count <= nb_neighbors:
        raise ValueError(
            "Too few points remain for the selected nb_neighbors. "
            "Reduce nb_neighbors or use a smaller voxel_size."
        )

    cleaned, inlier_indices = (
        downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
    )

    cleaned_count = len(cleaned.points)

    # Step 3: Save the cleaned point cloud.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_point_cloud(
        str(output_path),
        cleaned,
    )

    if not success:
        raise RuntimeError(
            f"Could not write output file: {output_path}"
        )

    removed_count = original_count - cleaned_count
    reduction_percent = (
        removed_count / original_count
    ) * 100

    print("\n--- Noise Reduction Summary ---")
    print(f"Original points:       {original_count:,}")
    print(f"After downsampling:    {downsampled_count:,}")
    print(f"Final cleaned points:  {cleaned_count:,}")
    print(f"Points removed:        {removed_count:,}")
    print(f"Total reduction:       {reduction_percent:.2f}%")
    print(f"Saved cleaned cloud:   {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample and denoise a PLY point cloud."
    )

    parser.add_argument(
        "--input",
        default="3d-cloud-projection/sharp_vacation_house.ply",
        help="Path to the input PLY file.",
    )

    parser.add_argument(
        "--output",
        default="3d-cloud-projection/cleaned_pointcloud.ply",
        help="Path to save the cleaned PLY file.",
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel size in the point cloud's coordinate units.",
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=20,
        help="Neighbour count for statistical outlier removal.",
    )

    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        help="Statistical outlier threshold multiplier.",
    )

    args = parser.parse_args()

    reduce_point_cloud_noise(
        input_file=args.input,
        output_file=args.output,
        voxel_size=args.voxel_size,
        nb_neighbors=args.neighbors,
        std_ratio=args.std_ratio,
    )

#!/bin/bash
#SBATCH --job-name=nuimg_setup
#SBATCH --output=/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/slurm_logs/nuimg_setup_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=a40
#SBATCH --account=agipml
#SBATCH --qos=a40

# ============================================================
# One-time setup: extract nuImages tarball and create symlinks
# into YOLO-UniOW JPEGImages/nuOWODB/ with timestamp-only names.
#
# Source images:   HONDA/ovow/data/nuscenes/nuimages-v1.0-all-samples.tgz
# Extracted to:    HONDA/ovow/data/nuscenes/samples/CAM_*/
# Symlinked to:    hypyolo/YOLO-UniOW/data/OWOD/JPEGImages/nuOWODB/
#
# Naming: n003-2018-01-02__CAM_FRONT__1514952221190440.jpg  →  1514952221190440.jpg
# ============================================================

set -e

NUSCENES_DIR="/home/agipml/sourav.rout/ALL_FILES/HONDA/ovow/data/nuscenes"
TARBALL="${NUSCENES_DIR}/nuimages-v1.0-all-samples.tgz"
DEST_DIR="/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/data/OWOD/JPEGImages/nuOWODB"

echo "[1/3] Checking tarball..."
if [[ ! -f "$TARBALL" ]]; then
    echo "ERROR: Tarball not found at $TARBALL"
    exit 1
fi

# Only extract if samples dirs are still empty
SAMPLE_COUNT=$(find "${NUSCENES_DIR}/samples" -name "*.jpg" | wc -l)
if [[ "$SAMPLE_COUNT" -eq 0 ]]; then
    echo "[1/3] Extracting tarball (16GB) to ${NUSCENES_DIR} ..."
    tar xzf "$TARBALL" -C "$NUSCENES_DIR"
    echo "[1/3] Extraction complete."
else
    echo "[1/3] Images already extracted ($SAMPLE_COUNT .jpg files found). Skipping."
fi

echo "[2/3] Creating symlinks in ${DEST_DIR} ..."
mkdir -p "$DEST_DIR"

LINKED=0
SKIPPED=0

# Images are nested as: samples/CAM_*/PREFIX__CAM_NAME__TIMESTAMP.jpg
# We want symlinks: TIMESTAMP.jpg -> /full/path/to/original.jpg
find "${NUSCENES_DIR}/samples" -name "*.jpg" | while read -r imgpath; do
    filename=$(basename "$imgpath")
    # Extract timestamp: last __-separated component without extension
    timestamp="${filename##*__}"
    timestamp="${timestamp%.jpg}"
    dest="${DEST_DIR}/${timestamp}.jpg"
    if [[ ! -e "$dest" ]]; then
        ln -s "$imgpath" "$dest"
        LINKED=$((LINKED + 1))
    else
        SKIPPED=$((SKIPPED + 1))
    fi
done

echo "[2/3] Symlink creation done."

echo "[3/3] Verification..."
IMG_COUNT=$(find "$DEST_DIR" -maxdepth 1 -type l | wc -l)
ANN_COUNT=$(find "/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/data/OWOD/Annotations/nuOWODB" -maxdepth 1 -type l | wc -l)
echo "  JPEGImages/nuOWODB symlinks: $IMG_COUNT"
echo "  Annotations/nuOWODB symlinks: $ANN_COUNT"

# Cross-check: verify all test split images are present
IMAGESET="/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/data/OWOD/ImageSets/nuOWODB/test.txt"
MISSING=0
while read -r img_id; do
    if [[ ! -e "${DEST_DIR}/${img_id}.jpg" ]]; then
        MISSING=$((MISSING + 1))
    fi
done < "$IMAGESET"
echo "  Test split missing images: $MISSING / $(wc -l < $IMAGESET)"

echo "[3/3] Setup complete."

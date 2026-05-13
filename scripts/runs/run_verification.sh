#!/bin/bash




set -e

OUTDIR="outputs/verification_run"
CSVDIR="${OUTDIR}/csv"

mkdir -p "${CSVDIR}"

echo "========================================================================"
echo "  Verification run started — 9 simulations, ~45 min total"
echo "  Output folder: ${OUTDIR}"
echo "========================================================================"


echo ""
echo "[1/9] base_abundant"
python -m sheep_sim --scenario abundant --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/base_abundant" \
  --disable-foraging --disable-social --disable-circadian \
  --disable-memory --disable-personality --disable-terrain
cp "${OUTDIR}/base_abundant/metrics.csv" "${CSVDIR}/metrics_base_abundant.csv"


echo ""
echo "[2/9] base_scarce"
python -m sheep_sim --scenario scarce --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/base_scarce" \
  --disable-foraging --disable-social --disable-circadian \
  --disable-memory --disable-personality --disable-terrain
cp "${OUTDIR}/base_scarce/metrics.csv" "${CSVDIR}/metrics_base_scarce.csv"


echo ""
echo "[3/9] iso_foraging (scarce)"
python -m sheep_sim --scenario scarce --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/iso_foraging_scarce" \
  --disable-social --disable-circadian \
  --disable-memory --disable-personality --disable-terrain
cp "${OUTDIR}/iso_foraging_scarce/metrics.csv" "${CSVDIR}/metrics_iso_foraging_scarce.csv"


echo ""
echo "[4/9] iso_memory (scarce)"
python -m sheep_sim --scenario scarce --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/iso_memory_scarce" \
  --disable-foraging --disable-social --disable-circadian \
  --disable-personality --disable-terrain
cp "${OUTDIR}/iso_memory_scarce/metrics.csv" "${CSVDIR}/metrics_iso_memory_scarce.csv"


echo ""
echo "[5/9] iso_social (abundant)"
python -m sheep_sim --scenario abundant --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/iso_social_abundant" \
  --disable-foraging --disable-circadian \
  --disable-memory --disable-personality --disable-terrain
cp "${OUTDIR}/iso_social_abundant/metrics.csv" "${CSVDIR}/metrics_iso_social_abundant.csv"


echo ""
echo "[6/9] iso_circadian (abundant)"
python -m sheep_sim --scenario abundant --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/iso_circadian_abundant" \
  --disable-foraging --disable-social \
  --disable-memory --disable-personality --disable-terrain
cp "${OUTDIR}/iso_circadian_abundant/metrics.csv" "${CSVDIR}/metrics_iso_circadian_abundant.csv"


echo ""
echo "[7/9] iso_personality (abundant)"
python -m sheep_sim --scenario abundant --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/iso_personality_abundant" \
  --disable-foraging --disable-social --disable-circadian \
  --disable-memory --disable-terrain
cp "${OUTDIR}/iso_personality_abundant/metrics.csv" "${CSVDIR}/metrics_iso_personality_abundant.csv"


echo ""
echo "[8/9] iso_terrain (abundant)"
python -m sheep_sim --scenario abundant --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/iso_terrain_abundant" \
  --disable-foraging --disable-social --disable-circadian \
  --disable-memory --disable-personality
cp "${OUTDIR}/iso_terrain_abundant/metrics.csv" "${CSVDIR}/metrics_iso_terrain_abundant.csv"


echo ""
echo "[9/9] full (abundant)"
python -m sheep_sim --scenario abundant --steps 1920 --seed 42 --landscape-seed 7 \
  --render none --output-dir "${OUTDIR}/full_abundant"
cp "${OUTDIR}/full_abundant/metrics.csv" "${CSVDIR}/metrics_full_abundant.csv"

echo ""
echo "========================================================================"
echo "  All 9 runs complete."
echo "  Renamed metrics files in: ${CSVDIR}/"
echo "========================================================================"
ls -lh "${CSVDIR}/"
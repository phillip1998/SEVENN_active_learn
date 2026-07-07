#!/bin/bash


#$ -pe mpi_36 36
#$ -N Gau_JYP
#$ -S /bin/bash
#$ -q all.q@node03
#$ -V
#$ -cwd



PROGRAM_ARGS=()   
if [[ -n "${OUTDIR:-}" ]]; then
  mkdir -p "$OUTDIR"
fi

inp_files=( *.gjf )

if (( ${#inp_files[@]} == 0 )); then
  echo "No .gjf files found in: $(pwd)"
  exit 0
fi

echo "Found ${#inp_files[@]} .gjf file(s)."

for inp in "${inp_files[@]}"; do
  base="${inp%.gjf}"
  out="${base}.log"
  err="${base}.err"
  if [[ -f "$out" ]]; then
    echo "Skip: $inp (already exists: $out)"
    continue
  fi
  echo "Running: g16 ${PROGRAM_ARGS[*]} $inp"
  g16 "$inp" >> "$out" 
done

echo "Done."

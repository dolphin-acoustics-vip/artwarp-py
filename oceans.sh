#!/usr/bin/env bash
#
# oceans.sh — Interactive OCEANS data-fetch launcher for artwarp-py
#
# Downloads dolphin call selections from the OCEANS API, extracts frequency
# contours from each WAV file, and saves one CSV per selection; ready for
# 'artwarp-py train'.
#
# OCEANS (Odontocete Call Environment and Archival Network) is developed by
# James Sullivan: https://github.com/dolphin-acoustics-vip/database-management-system
#
# Credentials are read from environment variables (!!!NEVER stored in this file).
# Set before running:
#   export OCEAN_USERNAME="your@email.ac.uk"
#   export OCEAN_PASSWORD="your_password"
#   # or, with a pre-obtained token:
#   export OCEAN_ACCESS_TOKEN="eyJ..."
#
# Test server (no API-privileges required; but not sure how extensive the test server is...):
#   export OCEAN_BASE_URL="https://rescomp-test-2.st-andrews.ac.uk/ocean/api"
#
# Usage:
#   ./oceans.sh              # interactive menu
#   ./oceans.sh fetch -o ./contours_ocean --max-per-species 20
#   ./oceans.sh count
#
# @author: Pedro Gronda Garrigues
set -e

# ---------- Colors and formatting ----------
if [[ -t 1 ]]; then
  BOLD="$(tput bold)"
  DIM="$(tput dim)"
  RED="$(tput setaf 1)"
  GREEN="$(tput setaf 2)"
  YELLOW="$(tput setaf 3)"
  CYAN="$(tput setaf 6)"
  MAGENTA="$(tput setaf 5)"
  RESET="$(tput sgr0)"
else
  BOLD= DIM= RED= GREEN= YELLOW= CYAN= MAGENTA= RESET=
fi

header()   { echo ""; echo "${BOLD}${CYAN}═══ $* ═══${RESET}"; echo ""; }
subheader(){ echo "${BOLD}$*${RESET}"; }
prompt()   { echo -n "${DIM}[${RESET} $* ${DIM}]${RESET} "; }
success()  { echo "${GREEN}$*${RESET}"; }
warn()     { echo "${YELLOW}$*${RESET}"; }
err()      { echo "${RED}$*${RESET}"; }
info()     { echo "${MAGENTA}$*${RESET}"; }

# ---------- Resolve artwarp-py executable ----------
ARTWARP_CMD=""
if command -v artwarp-py &>/dev/null; then
  ARTWARP_CMD="artwarp-py"
elif [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/artwarp-py" ]]; then
  ARTWARP_CMD="${VIRTUAL_ENV}/bin/artwarp-py"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
  if [[ -d "${SCRIPT_DIR}/src" ]] && [[ -f "${SCRIPT_DIR}/src/artwarp/__init__.py" ]]; then
    ARTWARP_CMD="python -m artwarp.cli.main"
  else
    ARTWARP_CMD="artwarp-py"
  fi
fi

# ---------- Input helpers ----------
prompt_with_default() {
  local label="$1" default="$2" validator="${3:-}"
  local val
  while true; do
    prompt "$label [${default}]" >&2
    read -r val
    val="${val:-$default}"
    case "$validator" in
      int)
        if [[ -n "$val" ]] && [[ ! "$val" =~ ^-?[0-9]+$ ]]; then
          err "Invalid integer. Try again." >&2; continue
        fi ;;
      float)
        if [[ -n "$val" ]] && [[ ! "$val" =~ ^-?[0-9]+\.?[0-9]*$ ]] && \
           [[ ! "$val" =~ ^-?[0-9]*\.[0-9]+$ ]]; then
          err "Invalid number. Try again." >&2; continue
        fi ;;
      optional_int)
        if [[ -n "$val" ]] && [[ ! "$val" =~ ^[0-9]+$ ]]; then
          err "Must be a positive integer, or leave empty for no limit." >&2; continue
        fi ;;
      quantile)
        if [[ -n "$val" ]]; then
          if [[ ! "$val" =~ ^[0-9]*\.?[0-9]+$ ]]; then
            err "Invalid number." >&2; continue
          fi
          if ! awk -v v="$val" 'BEGIN{exit !(v>=0&&v<=1)}' 2>/dev/null; then
            err "Quantile must be between 0.0 and 1.0." >&2; continue
          fi
        fi ;;
    esac
    echo "$val"
    return 0
  done
}

prompt_yesno() {
  local label="$1" default="${2:-n}"
  local val
  prompt "$label (y/n) [${default}]" >&2
  read -r val
  val="${val:-$default}"
  [[ "$val" =~ ^[yY] ]] && return 0 || return 1
}

prompt_output_dir() {
  local label="$1" default="$2"
  local val
  while true; do
    val=$(prompt_with_default "$label" "$default" path)
    [[ -z "$val" ]] && { err "Output directory required." >&2; continue; }
    echo "$val"
    return 0
  done
}

# ---------- Credential check ----------
check_credentials() {
  local has_creds=false

  if [[ -n "${OCEAN_ACCESS_TOKEN:-}" ]]; then
    info "  Credentials: OCEAN_ACCESS_TOKEN (pre-obtained token)" >&2
    has_creds=true
  elif [[ -n "${OCEAN_USERNAME:-}" ]] && [[ -n "${OCEAN_PASSWORD:-}" ]]; then
    info "  Credentials: OCEAN_USERNAME / OCEAN_PASSWORD (from environment)" >&2
    has_creds=true
  fi

  if [[ "$has_creds" == "false" ]]; then
    warn "  No OCEANS credentials found in environment." >&2
    warn "  Set environment variables before running:" >&2
    warn "    export OCEAN_USERNAME='your@email.ac.uk'" >&2
    warn "    export OCEAN_PASSWORD='your_password'" >&2
    warn "  Or pre-obtain a token:" >&2
    warn "    export OCEAN_ACCESS_TOKEN='eyJ...'" >&2
    warn "  (Tokens expire after ~30 minutes)" >&2
    echo "" >&2
    if ! prompt_yesno "Continue anyway (artwarp-py will prompt interactively)?" "y"; then
      err "Cancelled." >&2
      exit 1
    fi
  fi

  if [[ -n "${OCEAN_BASE_URL:-}" ]]; then
    info "  API server: ${OCEAN_BASE_URL}" >&2
  else
    info "  API server: production (https://research.st-andrews.ac.uk/ocean/api)" >&2
    warn "  To use the test server: export OCEAN_BASE_URL='https://rescomp-test-2.st-andrews.ac.uk/ocean/api'" >&2
  fi
}

# ---------- Fetch command ----------
run_fetch() {
  header "OCEANS → artwarp-py contour fetch"

  subheader "Credentials"
  check_credentials

  echo "" >&2
  subheader "Output"
  local output_dir
  output_dir=$(prompt_output_dir "Output directory for contour CSVs" "./contours_ocean")

  subheader "Species"
  echo "  Default: built-in bottlenose dolphin species IDs" >&2
  echo "  You can override with specific OCEANS species UUIDs." >&2
  local custom_species=()
  if prompt_yesno "Specify custom species UUID(s)?" "n"; then
    while true; do
      local sid
      prompt "Species UUID (empty to finish)" >&2
      read -r sid
      [[ -z "$sid" ]] && break
      custom_species+=("$sid")
    done
  fi

  subheader "Limits"
  local max_per_species
  max_per_species=$(prompt_with_default "Max contours per species (empty = no limit)" "" optional_int)

  subheader "Contour extraction"
  echo "  OCEANS WAVs are recorded at 500 kHz.  'auto' nperseg targets ≈50 Hz/bin resolution." >&2
  local nperseg peak_quantile freq_low freq_high
  nperseg=$(prompt_with_default "Spectrogram segment length (nperseg, empty=auto)" "" optional_int)
  peak_quantile=$(prompt_with_default "Noise-suppression quantile (0.0–1.0)" "0.9" quantile)
  echo "  Frequency bounds: by default, each selection's annotated band from OCEANS is used." >&2
  freq_low=$(prompt_with_default "Global freq low bound Hz (empty=from OCEANS metadata)" "" optional_int)
  freq_high=$(prompt_with_default "Global freq high bound Hz (empty=from OCEANS metadata)" "" optional_int)

  subheader "Output"
  local quiet=false
  if prompt_yesno "Quiet mode (suppress progress)?" "n"; then quiet=true; fi

  # Build argument list
  local args=("oceans" "fetch" "-o" "$output_dir")
  for sid in "${custom_species[@]}"; do
    args+=("--species-id" "$sid")
  done
  [[ -n "$max_per_species" ]] && args+=("--max-per-species" "$max_per_species")
  [[ -n "$nperseg" ]] && args+=("--nperseg" "$nperseg")
  args+=("--peak-quantile" "$peak_quantile")
  [[ -n "$freq_low" ]]  && args+=("--freq-low"  "$freq_low")
  [[ -n "$freq_high" ]] && args+=("--freq-high" "$freq_high")
  [[ "$quiet" == "true" ]] && args+=("--quiet")

  echo "" >&2
  header "Command"
  echo "  $ARTWARP_CMD ${args[*]}" >&2
  echo "" >&2

  if prompt_yesno "Run this command?" "y"; then
    if $ARTWARP_CMD "${args[@]}"; then
      success "Done."
      echo "" >&2
      success "Contour CSVs written to: $output_dir" >&2
      info "Next step: train an ARTwarp network on the fetched data:" >&2
      echo "  artwarp-py train --input-dir $output_dir --format csv" >&2
      info "Or run the interactive launcher: ./run.sh" >&2
    else
      err "Fetch failed (exit $?)."
      return 1
    fi
  else
    warn "Cancelled." >&2
  fi
}

# ---------- Count command ----------
run_count() {
  header "OCEANS selection count"

  subheader "Credentials"
  check_credentials

  echo "" >&2
  subheader "Species"
  local custom_species=()
  if prompt_yesno "Specify custom species UUID(s)?" "n"; then
    while true; do
      local sid
      prompt "Species UUID (empty to finish)" >&2
      read -r sid
      [[ -z "$sid" ]] && break
      custom_species+=("$sid")
    done
  fi

  local args=("oceans" "count")
  for sid in "${custom_species[@]}"; do
    args+=("--species-id" "$sid")
  done

  echo "" >&2
  header "Command"
  echo "  $ARTWARP_CMD ${args[*]}" >&2
  echo "" >&2

  if prompt_yesno "Run this command?" "y"; then
    if $ARTWARP_CMD "${args[@]}"; then
      success "Done."
    else
      err "Count failed (exit $?)."
      return 1
    fi
  else
    warn "Cancelled." >&2
  fi
}

# ---------- Main menu ----------
# Writes the validated choice into the global MENU_CHOICE variable so callers
# do not need command substitution (which would capture all stdout, including
# output from header/info/success helpers, corrupting the return value, big no no).
MENU_CHOICE=""
main_menu() {
  local choice
  while true; do
    header "OCEANS data pipeline for artwarp-py"
    echo "  OCEANS (Odontocete Call Environment and Archival Network)"
    echo "  Developed by James Sullivan — https://github.com/dolphin-acoustics-vip/database-management-system"
    echo ""
    echo "  1) Fetch   — Download selections, extract contours to CSV"
    echo "  2) Count   — Count available selections (no download)"
    echo "  3) Quit"
    echo ""
    prompt "Choose 1–3"
    read -r choice
    choice="${choice// /}"
    case "$choice" in
      1|2|3) MENU_CHOICE="$choice"; return 0 ;;
    esac
    err "Invalid option. Enter 1, 2, or 3."
  done
}

# ---------- Entry point ----------
main() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
  if [[ -d "${SCRIPT_DIR}/src" ]] && [[ -f "${SCRIPT_DIR}/src/artwarp/__init__.py" ]]; then
    cd "$SCRIPT_DIR"
  fi

  # non-interactive: forward to artwarp-py oceans ...
  if [[ $# -ge 1 ]]; then
    $ARTWARP_CMD oceans "$@"
    exit $?
  fi

  while true; do
    main_menu
    case "$MENU_CHOICE" in
      1) run_fetch ;;
      2) run_count ;;
      3) success "Bye."; exit 0 ;;
    esac
    echo ""
    prompt "Press Enter to continue..."
    read -r
  done
}

main "$@"

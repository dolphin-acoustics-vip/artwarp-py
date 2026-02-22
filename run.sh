#!/usr/bin/env bash
#
# ARTwarp-py — interactive launcher and CLI entry point
#
# Interactive (no args): menu to configure and run train / plot / predict / export
#   with prompts for every CLI option, defaults, and validation.
# Direct (with args): forwards to artwarp-py (e.g. ./run.sh train -i ./contours -o results.pkl).
#
# Usage:
#   ./run.sh              # interactive menu
#   ./run.sh train -i ./contours -o results.pkl --vigilance 85
#   ./run.sh plot -r results.pkl -i ./contours -o ./report
#
# Run from repo root or anywhere. Uses artwarp-py on PATH or python -m artwarp.cli.main.
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
  BLUE="$(tput setaf 4)"
  CYAN="$(tput setaf 6)"
  RESET="$(tput sgr0)"
else
  BOLD= DIM= RED= GREEN= YELLOW= BLUE= CYAN= RESET=
fi

header()   { echo ""; echo "${BOLD}${CYAN}═══ $* ═══${RESET}"; echo ""; }
subheader(){ echo "${BOLD}$*${RESET}"; }
prompt()   { echo -n "${DIM}[${RESET} $* ${DIM}]${RESET} "; }
success()  { echo "${GREEN}$*${RESET}"; }
warn()     { echo "${YELLOW}$*${RESET}"; }
err()      { echo "${RED}$*${RESET}"; }

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
# prompt_with_default "Prompt text" "default_value" ["validator"]
# validator: int | float | path | optional_int | vigilance | learning_rate_01 | bias_01 | int_positive
# reprompts until valid (except path which is not validated here).
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
          err "Invalid integer. Try again." >&2
          continue
        fi
        ;;
      float)
        if [[ -n "$val" ]] && [[ ! "$val" =~ ^-?[0-9]+\.?[0-9]*$ ]] && [[ ! "$val" =~ ^-?[0-9]*\.[0-9]+$ ]]; then
          err "Invalid number. Try again." >&2
          continue
        fi
        ;;
      path) ;;
      optional_int)
        if [[ -n "$val" ]] && [[ ! "$val" =~ ^-?[0-9]+$ ]]; then
          err "Invalid integer (or leave empty). Try again." >&2
          continue
        fi
        ;;
      vigilance)
        if [[ -n "$val" ]]; then
          if [[ ! "$val" =~ ^-?[0-9]+\.?[0-9]*$ ]] && [[ ! "$val" =~ ^-?[0-9]*\.[0-9]+$ ]]; then
            err "Invalid number. Try again." >&2
            continue
          fi
          if ! awk -v v="$val" 'BEGIN{exit !(v>=1&&v<=99)}' 2>/dev/null; then
            err "Vigilance must be between 1 and 99. Try again." >&2
            continue
          fi
        fi
        ;;
      learning_rate_01|bias_01)
        if [[ -n "$val" ]]; then
          if [[ ! "$val" =~ ^-?[0-9]+\.?[0-9]*$ ]] && [[ ! "$val" =~ ^-?[0-9]*\.[0-9]+$ ]]; then
            err "Invalid number. Try again." >&2
            continue
          fi
          if ! awk -v v="$val" 'BEGIN{exit !(v>=0&&v<=1)}' 2>/dev/null; then
            err "Value must be between 0 and 1. Try again." >&2
            continue
          fi
        fi
        ;;
      int_positive)
        if [[ -n "$val" ]]; then
          if [[ ! "$val" =~ ^[0-9]+$ ]]; then
            err "Must be a positive integer. Try again." >&2
            continue
          fi
          if [[ "$val" -lt 1 ]]; then
            err "Must be at least 1. Try again." >&2
            continue
          fi
        fi
        ;;
    esac
    echo "$val"
    return 0
  done
}

# prompt -> for existing directory; reprompt until it exists
prompt_input_dir() {
  local label="$1" default="$2"
  local val
  while true; do
    val=$(prompt_with_default "$label" "$default" path)
    [[ -z "$val" ]] && { err "Input directory required." >&2; continue; }
    if [[ ! -d "$val" ]]; then
      err "Directory does not exist: $val. Try again." >&2
      continue
    fi
    echo "$val"
    return 0
  done
}

# prompt -> for output .pkl; if file exists, ask overwrite; reprompt if they decline
prompt_output_pkl() {
  local label="$1" default="$2"
  local val
  while true; do
    val=$(prompt_with_default "$label" "$default" path)
    [[ -z "$val" ]] && { err "Output file required." >&2; continue; }
    if [[ -f "$val" ]]; then
      warn "File already exists: $val" >&2
      if prompt_yesno "Overwrite?" "n"; then
        echo "$val"
        return 0
      fi
      continue
    fi
    echo "$val"
    return 0
  done
}

# prompt -> for existing .pkl file (results/model); reprompt until it exists
prompt_results_pkl() {
  local label="$1" default="$2"
  local val
  while true; do
    val=$(prompt_with_default "$label" "$default" path)
    [[ -z "$val" ]] && { err "Results/model file required." >&2; continue; }
    if [[ ! -f "$val" ]]; then
      err "File does not exist: $val. Try again." >&2
      continue
    fi
    echo "$val"
    return 0
  done
}

# prompt -> for output file (e.g. CSV); if file exists, ask overwrite; reprompt if they decline
prompt_output_file() {
  local label="$1" default="$2"
  local val
  while true; do
    val=$(prompt_with_default "$label" "$default" path)
    [[ -z "$val" ]] && { err "Output file required." >&2; continue; }
    if [[ -f "$val" ]]; then
      warn "File already exists: $val" >&2
      if prompt_yesno "Overwrite?" "n"; then
        echo "$val"
        return 0
      fi
      continue
    fi
    echo "$val"
    return 0
  done
}

# prompt -> for format (auto|ctr|csv|txt); reprompt until valid
prompt_format() {
  local default="${1:-auto}"
  local val
  while true; do
    echo "  1) auto  2) ctr  3) csv  4) txt" >&2
    val=$(prompt_with_default "Format (1–4 or name)" "$default" path)
    case "$val" in 1|auto) echo "auto"; return 0 ;; 2|ctr) echo "ctr"; return 0 ;; 3|csv) echo "csv"; return 0 ;; 4|txt) echo "txt"; return 0 ;; esac
    err "Invalid format. Choose 1, 2, 3, 4, or auto/ctr/csv/txt. Try again." >&2
  done
}

# prompt -> for export type (all|references|categories); reprompt until valid
prompt_export_type() {
  local default="${1:-all}"
  local val
  while true; do
    echo "  1) all  2) references  3) categories" >&2
    val=$(prompt_with_default "Export type" "$default" path)
    case "$val" in 1|all) echo "all"; return 0 ;; 2|references) echo "references"; return 0 ;; 3|categories) echo "categories"; return 0 ;; esac
    err "Invalid export type. Choose 1, 2, 3, or all/references/categories. Try again." >&2
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

# ---------- Main menu ----------
# menu -> printed to stderr; reprompt until valid choice 1–5
main_menu() {
  local choice
  while true; do
    header "ARTwarp-py" >&2
    echo "  1) Train   — Train network on contour directory, save model (.pkl)" >&2
    echo "  2) Plot    — Generate visualization report from results (.pkl)" >&2
    echo "  3) Predict — Predict categories for new contours using trained model" >&2
    echo "  4) Export  — Export reference contours / category assignments from .pkl" >&2
    echo "  5) Quit" >&2
    echo "" >&2
    prompt "Choose 1–5" >&2
    read -r choice
    choice="${choice// /}"
    case "$choice" in 1|2|3|4|5) echo "$choice"; return 0 ;; esac
    err "Invalid option. Enter 1, 2, 3, 4, or 5." >&2
  done
}

# ---------- Train ----------
run_train() {
  header "Train ARTwarp network"
  local input_dir output format
  local vigilance learning_rate bias max_categories max_iterations warp_factor seed
  local resample sample_interval tempres max_contour_length
  local export_refs export_categories quiet
  local args=()

  subheader "Paths"
  input_dir=$(prompt_input_dir "Input directory (contour files)" "./contours")
  output=$(prompt_output_pkl "Output model file (.pkl)" "results.pkl")
  args+=(-i "$input_dir" -o "$output")

  subheader "Input format"
  format=$(prompt_format "auto")
  args+=(--format "$format")
  # frequency column -> not prompted; CLI default 0 is used. CSV loader auto-detects Hz/Frequency column when possible

  subheader "Network parameters"
  vigilance=$(prompt_with_default "Vigilance (1–99)" "85.0" vigilance)
  args+=(--vigilance "$vigilance")
  learning_rate=$(prompt_with_default "Learning rate (0–1)" "0.1" learning_rate_01)
  args+=(--learning-rate "$learning_rate")
  bias=$(prompt_with_default "Bias (0–1)" "0.0" bias_01)
  args+=(--bias "$bias")
  max_categories=$(prompt_with_default "Max categories" "50" int_positive)
  args+=(--max-categories "$max_categories")
  max_iterations=$(prompt_with_default "Max iterations" "50" int_positive)
  args+=(--max-iterations "$max_iterations")
  warp_factor=$(prompt_with_default "Warp factor (DTW)" "3" int_positive)
  args+=(--warp-factor "$warp_factor")
  seed=$(prompt_with_default "Random seed (empty = none)" "" optional_int)
  [[ -n "$seed" ]] && args+=(--seed "$seed")

  subheader "Resampling (optional)"
  if prompt_yesno "Resample contours to uniform temporal resolution?" "n"; then
    args+=(--resample)
    sample_interval=$(prompt_with_default "Sample interval (seconds)" "0.02" float)
    args+=(--sample-interval "$sample_interval")
    tempres=$(prompt_with_default "Default tempres (sec/point)" "0.01" float)
    args+=(--tempres "$tempres")
  fi
  max_contour_length=$(prompt_with_default "Max contour length (empty = no cap)" "" optional_int)
  [[ -n "$max_contour_length" ]] && args+=(--max-contour-length "$max_contour_length")

  subheader "Output options"
  if prompt_yesno "Export reference contours to CSV?" "n"; then args+=(--export-refs); fi
  if prompt_yesno "Export category assignments to CSV?" "n"; then args+=(--export-categories); fi
  if prompt_yesno "Quiet (no progress)?" "n"; then args+=(--quiet); fi

  echo ""
  header "Command"
  echo "  $ARTWARP_CMD train ${args[*]}"
  echo ""
  if prompt_yesno "Run this command?" "y"; then
    if $ARTWARP_CMD train "${args[@]}"; then
      success "Done."
    else
      err "Command failed (exit $?)."
      return 1
    fi
  else
    warn "Cancelled."
  fi
}

# ---------- Plot ----------
run_plot() {
  header "Generate visualization report"
  local results input_dir output_dir format dpi
  local args=()

  subheader "Paths"
  results=$(prompt_results_pkl "Results file (.pkl)" "results.pkl")
  args+=(-r "$results")
  input_dir=$(prompt_input_dir "Input directory (contour files)" "./contours")
  args+=(-i "$input_dir")
  output_dir=$(prompt_with_default "Output directory for figures" "./report" path)
  [[ -z "$output_dir" ]] && output_dir="./report"
  args+=(-o "$output_dir")

  subheader "Input format"
  format=$(prompt_format "auto")
  args+=(--format "$format")
  dpi=$(prompt_with_default "DPI for figures" "300" int_positive)
  args+=(--dpi "$dpi")

  echo ""
  header "Command"
  echo "  $ARTWARP_CMD plot ${args[*]}"
  echo ""
  if prompt_yesno "Run this command?" "y"; then
    if $ARTWARP_CMD plot "${args[@]}"; then
      success "Done."
    else
      err "Command failed (exit $?)."
      return 1
    fi
  else
    warn "Cancelled."
  fi
}

# ---------- Predict ----------
run_predict() {
  header "Predict categories"
  local model input_dir output format quiet
  local args=()

  subheader "Paths"
  model=$(prompt_results_pkl "Trained model (.pkl)" "results.pkl")
  args+=(-m "$model")
  input_dir=$(prompt_input_dir "Input directory (contours to predict)" "./contours")
  args+=(-i "$input_dir")
  output=$(prompt_output_file "Output CSV file" "predictions.csv")
  args+=(-o "$output")

  subheader "Input format"
  format=$(prompt_format "auto")
  args+=(--format "$format")
  # frequency column -> CLI default 0; CSV loader auto-detects Hz/Frequency column when possible
  if prompt_yesno "Quiet?" "n"; then args+=(--quiet); fi

  echo ""
  header "Command"
  echo "  $ARTWARP_CMD predict ${args[*]}"
  echo ""
  if prompt_yesno "Run this command?" "y"; then
    if $ARTWARP_CMD predict "${args[@]}"; then
      success "Done."
    else
      err "Command failed (exit $?)."
      return 1
    fi
  else
    warn "Cancelled."
  fi
}

# ---------- Export ----------
run_export() {
  header "Export results"
  local results output_dir export_type
  local args=()

  subheader "Paths"
  results=$(prompt_results_pkl "Results file (.pkl)" "results.pkl")
  args+=(-r "$results")
  output_dir=$(prompt_with_default "Output directory" "./export" path)
  [[ -z "$output_dir" ]] && { err "Output directory required."; return 1; }
  args+=(-o "$output_dir")
  export_type=$(prompt_export_type "all")
  args+=(--export-type "$export_type")

  echo ""
  header "Command"
  echo "  $ARTWARP_CMD export ${args[*]}"
  echo ""
  if prompt_yesno "Run this command?" "y"; then
    if $ARTWARP_CMD export "${args[@]}"; then
      success "Done."
    else
      err "Command failed (exit $?)."
      return 1
    fi
  else
    warn "Cancelled."
  fi
}

# ---------- Entry ----------
main() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
  if [[ -d "${SCRIPT_DIR}/src" ]] && [[ -f "${SCRIPT_DIR}/src/artwarp/__init__.py" ]]; then
    cd "$SCRIPT_DIR"
  fi

  # non-interactive -> forward all arguments to artwarp-py (e.g. ./run.sh train -i ./c -o out.pkl)
  if [[ $# -ge 1 ]]; then
    $ARTWARP_CMD "$@"
    exit $?
  fi

  while true; do
    choice=$(main_menu)
    case "$choice" in
      1) run_train ;;
      2) run_plot ;;
      3) run_predict ;;
      4) run_export ;;
      5) success "Bye."; exit 0 ;;
      *) err "Invalid option." ;;
    esac
    echo ""
    prompt "Press Enter to continue..."
    read -r
  done
}

main "$@"

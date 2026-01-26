#!/bin/bash
#
# TensorRT Engine Builder
#
# Build TensorRT engines from ONNX models using configuration files.
# Supports dynamic shapes, FP16/INT8 precision, NVDLA acceleration with GPU fallback.
#
# Usage:
#   ./build_engine.sh <config_file.json>
#   ./build_engine.sh                     # Interactive mode
#   ./build_engine.sh -h, --help          # Show help
#
# Author: TrtEngineToolkits
# Date: 2025-11-28
#

# ----------------------------
# Color codes and utilities
# ----------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${RESET} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${RESET} $1" >&2
}

print_header() {
    echo -e "${CYAN}==================================================${RESET}"
    echo -e "${CYAN}$1${RESET}"
    echo -e "${CYAN}==================================================${RESET}"
}

# ----------------------------
# JSON Parsing (using jq or Python fallback)
# ----------------------------
json_get() {
    local json_file="$1"
    local key="$2"

    if command -v jq &>/dev/null; then
        jq -r "$key // empty" "$json_file" 2>/dev/null
    else
        python3 -c "
import json, sys
try:
    with open('$json_file') as f:
        data = json.load(f)
    keys = '$key'.strip('.').split('.')
    result = data
    for k in keys:
        if k and k in result:
            result = result[k]
        else:
            sys.exit(0)
    if result is not None and result != '':
        print(result if not isinstance(result, bool) else str(result).lower())
except:
    pass
" 2>/dev/null
    fi
}

json_get_bool() {
    local json_file="$1"
    local key="$2"
    local default="${3:-false}"

    local value
    value=$(json_get "$json_file" "$key")

    if [[ -z "$value" ]]; then
        echo "$default"
    elif [[ "$value" == "true" ]] || [[ "$value" == "True" ]] || [[ "$value" == "1" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

json_get_int() {
    local json_file="$1"
    local key="$2"
    local default="${3:-0}"

    local value
    value=$(json_get "$json_file" "$key")

    if [[ -z "$value" ]] || ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# ----------------------------
# TensorRT Engine Building
# ----------------------------
run_trtexec() {
    local -a command=("$@")
    local attempt_name="${command[-1]}"
    unset 'command[-1]'  # Remove attempt name from command

    print_header "TensorRT Engine Building Initiated ($attempt_name)"
    echo "Generated Command: ${command[*]}"
    echo ""

    # Run trtexec and capture output
    local exit_code=0
    local stderr_file
    stderr_file=$(mktemp)

    # Execute and show output in real-time
    "${command[@]}" 2>"$stderr_file"
    exit_code=$?

    # Show stderr if any
    if [[ -s "$stderr_file" ]]; then
        print_warn "STDERR:"
        cat "$stderr_file"
    fi

    local stderr_content
    stderr_content=$(cat "$stderr_file")
    rm -f "$stderr_file"

    if [[ $exit_code -ne 0 ]]; then
        echo "$stderr_content"
        return 1
    fi

    return 0
}

build_tensorrt_engine() {
    local config_file="$1"

    # Validate config file
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file '$config_file' not found."
        return 1
    fi

    # Check if it's valid JSON
    if ! python3 -c "import json; json.load(open('$config_file'))" 2>/dev/null; then
        print_error "Could not decode JSON from '$config_file'. Please check the file's syntax."
        return 1
    fi

    # Extract paths
    local onnx_path
    local target_path
    onnx_path=$(json_get "$config_file" ".onnx_path")
    target_path=$(json_get "$config_file" ".target_path")

    if [[ -z "$onnx_path" ]] || [[ -z "$target_path" ]]; then
        print_error "'onnx_path' and 'target_path' must be specified in the config."
        return 1
    fi

    # Check if ONNX file exists
    if [[ ! -f "$onnx_path" ]]; then
        print_error "ONNX file not found: $onnx_path"
        return 1
    fi

    # Check if trtexec exists
    if ! command -v trtexec &>/dev/null; then
        print_error "'trtexec' command not found. Please ensure TensorRT is installed and configured in your PATH."
        return 1
    fi

    # Build base command
    local -a base_command=("trtexec")
    base_command+=("--onnx=$onnx_path")
    base_command+=("--saveEngine=$target_path")

    # Extract trt_config options
    local precision
    precision=$(json_get "$config_file" ".trt_config.precision")
    precision="${precision:-fp32}"

    case "$precision" in
        fp16)
            base_command+=("--fp16")
            ;;
        int8)
            base_command+=("--int8")
            print_warn "INT8 precision selected. Full INT8 calibration setup is not included in this basic script."
            ;;
    esac

    # Add dynamic shapes
    local min_shapes opt_shapes max_shapes
    min_shapes=$(json_get "$config_file" ".trt_config.min_shapes")
    opt_shapes=$(json_get "$config_file" ".trt_config.opt_shapes")
    max_shapes=$(json_get "$config_file" ".trt_config.max_shapes")

    if [[ -n "$min_shapes" ]] && [[ -n "$opt_shapes" ]] && [[ -n "$max_shapes" ]]; then
        base_command+=("--minShapes=$min_shapes")
        base_command+=("--optShapes=$opt_shapes")
        base_command+=("--maxShapes=$max_shapes")
    else
        print_warn "Dynamic shapes (min_shapes, opt_shapes, max_shapes) are recommended for flexible engines."
    fi

    # Add verbose flag
    local verbose
    verbose=$(json_get_bool "$config_file" ".trt_config.verbose" "false")
    if [[ "$verbose" == "true" ]]; then
        base_command+=("--verbose")
    fi

    # Add useSpinWait flag
    local use_spin_wait
    use_spin_wait=$(json_get_bool "$config_file" ".trt_config.use_spin_wait" "false")
    if [[ "$use_spin_wait" == "true" ]]; then
        base_command+=("--useSpinWait")
        print_info "--useSpinWait is enabled, which may increase CPU utilization but reduce latency."
    fi

    # Print build info
    print_header "TensorRT Engine Build Configuration"
    echo "Configuration file: $config_file"
    echo "Input ONNX: $onnx_path"
    echo "Output Engine: $target_path"
    echo "Precision: $precision"

    # --- NVDLA Attempt Logic ---
    local use_nvdla
    local nvdla_core_id
    use_nvdla=$(json_get_bool "$config_file" ".trt_config.use_nvdla" "false")
    nvdla_core_id=$(json_get_int "$config_file" ".trt_config.nvdla_core_id" "0")

    if [[ "$use_nvdla" == "true" ]]; then
        local -a nvdla_command=("${base_command[@]}")
        nvdla_command+=("--useDLACore=$nvdla_core_id")
        nvdla_command+=("--allowGPUFallback")

        print_info "NVDLA Enabled: Yes (Core $nvdla_core_id) with GPU fallback."
        echo ""

        # Run with NVDLA
        if run_trtexec "${nvdla_command[@]}" "Attempting with NVDLA"; then
            echo ""
            print_header "TensorRT Engine Built Successfully with NVDLA!"
            return 0
        else
            print_warn "NVDLA compilation failed. Attempting to build engine without NVDLA acceleration..."
            sleep 1
        fi
    else
        print_info "NVDLA Enabled: No"
    fi

    # --- GPU-only Fallback / Default Compilation ---
    echo ""
    print_info "Building with GPU only..."

    if run_trtexec "${base_command[@]}" "GPU Only"; then
        echo ""
        print_header "TensorRT Engine Built Successfully (GPU Only)!"
        return 0
    else
        print_error "trtexec command failed during GPU-only compilation."
        return 1
    fi
}

# ----------------------------
# Interactive Mode
# ----------------------------
list_configs() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local config_dir="${script_dir}/../config"

    if [[ ! -d "$config_dir" ]]; then
        print_warn "Config directory not found: $config_dir"
        return 1
    fi

    local -a configs
    mapfile -t configs < <(find "$config_dir" -name "*.json" 2>/dev/null | sort)

    if [[ ${#configs[@]} -eq 0 ]]; then
        print_warn "No JSON configuration files found in $config_dir"
        return 1
    fi

    print_info "Searching for configuration files in: $config_dir"
    echo ""
    echo -e "${GREEN}Available configuration files:${RESET}"
    for i in "${!configs[@]}"; do
        echo "  [$((i+1))] ${configs[$i]}"
    done
    echo ""

    # Return configs via global variable
    FOUND_CONFIGS=("${configs[@]}")
    return 0
}

run_interactive() {
    print_info "Interactive mode - No configuration file specified"
    echo ""

    if ! list_configs; then
        print_info "Please provide a configuration file as argument."
        echo ""
        show_usage
        return 1
    fi

    read -p "Enter the number of config file to use (or press Enter to exit): " selection

    if [[ -z "$selection" ]]; then
        print_info "No selection made. Exiting."
        return 0
    fi

    local index=$((selection - 1))

    if [[ $index -ge 0 ]] && [[ $index -lt ${#FOUND_CONFIGS[@]} ]]; then
        build_tensorrt_engine "${FOUND_CONFIGS[$index]}"
    else
        print_error "Invalid selection: $selection"
        return 1
    fi
}

# ----------------------------
# Show Example Config
# ----------------------------
show_example_config() {
    cat << 'EOF'
{
    "onnx_path": "/opt/models/efficientnet_b0_feat_logits.onnx",
    "target_path": "/opt/models/efficientnet_b0_feat_logits.engine",
    "model_information": {
        "name": "EfficientNet B0 Feature and Logits",
        "description": "EfficientNet B0 model for feature extraction and logits output",
        "version": "1.0.0",
        "author": "Orlando Chen",
        "date": "2023-10-01",
        "input_tensors": [
            {"name": "input", "dimensions": [1, 3, 224, 224]}
        ],
        "output_tensors": [
            {"name": "logits", "dimensions": [1, 2]},
            {"name": "feat", "dimensions": [1, 256]}
        ]
    },
    "trt_config": {
        "precision": "fp16",
        "min_shapes": "input:1x3x224x224",
        "opt_shapes": "input:8x3x224x224",
        "max_shapes": "input:32x3x224x224",
        "use_nvdla": true,
        "nvdla_core_id": 0,
        "verbose": true,
        "use_spin_wait": false
    }
}
EOF
}

# ----------------------------
# Help
# ----------------------------
show_usage() {
    cat << EOF
${GREEN}TensorRT Engine Builder${RESET}

${BLUE}Usage:${RESET}
  $0 <config_file.json>    Build engine from configuration file
  $0                       Interactive mode (browse and select config)
  $0 --example             Show example configuration file
  $0 -h, --help            Show this help message

${BLUE}Examples:${RESET}
  $0 config/yolov8_pose.json
  $0 ../models/efficientnet.json

${BLUE}Configuration File Format:${RESET}
  The JSON configuration file should contain:
    - onnx_path: Path to input ONNX model
    - target_path: Path for output TensorRT engine
    - trt_config: TensorRT build options
        - precision: "fp32", "fp16", or "int8"
        - min_shapes: Minimum input shapes for dynamic batching
        - opt_shapes: Optimal input shapes
        - max_shapes: Maximum input shapes
        - use_nvdla: Enable NVDLA acceleration (Jetson only)
        - nvdla_core_id: NVDLA core to use (0 or 1)
        - verbose: Enable verbose output
        - use_spin_wait: Enable spin wait for lower latency

${BLUE}Features:${RESET}
  - Automatic NVDLA detection and fallback to GPU
  - Dynamic shape support
  - FP16/INT8 precision options
  - Real-time build progress display
  - Comprehensive error handling

${BLUE}Requirements:${RESET}
  - TensorRT installed with trtexec in PATH
  - jq (optional, for JSON parsing) or Python3

EOF
}

# ----------------------------
# Main
# ----------------------------
main() {
    # Handle command line arguments
    case "${1:-}" in
        -h|--help)
            show_usage
            exit 0
            ;;
        --example)
            print_info "Example configuration file:"
            echo ""
            show_example_config
            exit 0
            ;;
        "")
            # Interactive mode
            run_interactive
            exit $?
            ;;
        *)
            # Configuration file provided
            local config_file="$1"

            # Check if file exists
            if [[ ! -f "$config_file" ]]; then
                print_error "Configuration file not found: $config_file"
                echo ""
                print_info "You can create a new config file based on this example:"
                show_example_config
                exit 1
            fi

            # Check if it's a JSON file
            if [[ ! "$config_file" =~ \.json$ ]]; then
                print_warn "File does not have .json extension: $config_file"
                read -p "Continue anyway? (y/N): " confirm
                if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                    print_info "Build cancelled."
                    exit 0
                fi
            fi

            build_tensorrt_engine "$config_file"
            exit $?
            ;;
    esac
}

main "$@"

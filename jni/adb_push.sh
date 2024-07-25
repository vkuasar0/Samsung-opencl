#!/bin/bash

# Define paths and variables
MODULE_PATH="../libs/arm64-v8a/myModule"
INPUT_IMGS_PATH="../inputImgs"
ASSETS_PATH="../assets"
CONFIG_FILE="./config.pbtx"
DEVICE_PATH="/data/local/tmp"
TRACE_FILE="trace_file.perfetto-trace"
PERFETTO_PATH="./perfetto/out/default/trace_processor_shell"

# Function to push a file and check for success
adb_push() {
    adb push "$1" "$2"
    if [ $? -ne 0 ]; then
        echo "Failed to push $1 to $2"
        exit 1
    fi
}

# Check if config.pbtx file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: $CONFIG_FILE does not exist."
  exit 1
fi

# Push the necessary files to the device
adb_push "$MODULE_PATH" "$DEVICE_PATH"
adb_push "$INPUT_IMGS_PATH/image_add1.png" "$DEVICE_PATH"
adb_push "$INPUT_IMGS_PATH/image_add2.png" "$DEVICE_PATH"
adb_push "$INPUT_IMGS_PATH/output1.png" "$DEVICE_PATH"
adb_push "$INPUT_IMGS_PATH/output2.png" "$DEVICE_PATH"
adb_push "$INPUT_IMGS_PATH/output3.png" "$DEVICE_PATH"
adb_push "$ASSETS_PATH/." "$DEVICE_PATH"
adb_push "$CONFIG_FILE" "$DEVICE_PATH/trace_config.pbtx"

# Execute the commands on the device
adb shell <<EOF
cd "$DEVICE_PATH"
# Start tracing with Perfetto using the config file
cat trace_config.pbtx | perfetto --txt -c - -o /data/misc/perfetto-traces/$TRACE_FILE &
PERFETTO_PID=\$!
# Run myModule
./myModule 0
# Stop tracing
kill \$PERFETTO_PID || true
exit
EOF

# Pull the trace file to the local machine
adb pull "/data/misc/perfetto-traces/$TRACE_FILE" .

# Debugging: Check if the Perfetto binary exists
if [ ! -f "$PERFETTO_PATH" ]; then
    echo "Perfetto binary could not be found at $PERFETTO_PATH. Please check the path."
    exit 1
else
    echo "Perfetto binary found at $PERFETTO_PATH."
fi

# Process the trace file locally
# Example SQL query to extract GPU usage
QUERY="SELECT SUM(value) AS gpu_usage FROM counter WHERE name LIKE '%gpu_busy%';"
gpu_usage=$("$PERFETTO_PATH" --query-string "$QUERY" "$TRACE_FILE" | awk '{print $NF}' | tail -n 1)
echo "GPU Usage: $gpu_usage"

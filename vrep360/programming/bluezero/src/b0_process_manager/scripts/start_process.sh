B0_CONSOLE_LOGLEVEL=warn ./b0_service_call -s process_manager/control -c b0::process_manager::Request <<EOF \
    | grep -v ^Content-type: \
    | python -m json.tool
{
    "start_process": {
        "path": "$1",
        "args": []
    }
}
EOF

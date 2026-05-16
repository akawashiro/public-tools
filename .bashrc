export PATH=$HOME/.cargo/bin:$PATH

if which renlog > /dev/null 2>&1; then
    if [[ "$(ps -o comm= -p $PPID)" != "renlog" ]]; then
        renlog_dir=/tmp/renlog
        exec renlog --log-level info log --renlog-dir ${renlog_dir} --cmd 'bash -l'
    else
        eval "$(renlog show-bash-rc)"
    fi
fi

renlog_view_last_cmd() {
    local last_log_file
    last_log_file=$(cat "${RENLOG_LAST_LOG_FILE}")
    if [ -f "${last_log_file}" ]; then
        nvim "${last_log_file}"
    else
        echo "No log file found."
    fi
}

bind -x '"\e1":renlog_view_last_cmd'

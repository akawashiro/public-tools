########## Environment variables start ##########

export LANG=en_US.UTF-8

export PATH=/var/lib/snapd/snap/bin:$PATH
export PATH=$HOME/.cabal/bin:$PATH
export PATH=$HOME/.local/bin:$PATH
export PATH=$HOME/.local/sbin:$PATH
export PATH=$HOME/public-tools:$PATH
export PATH=$HOME/tools:$PATH
export PATH=$HOME/.cargo/bin:$PATH
export PATH=$HOME/.gem/ruby/2.7.0/bin:$PATH
export PATH=$HOME/.fzf/bin:$PATH
export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"

export GOPATH="$HOME/.go"
export PATH="$GOPATH/bin:$PATH"

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"

# The next line updates PATH for the Google Cloud SDK.
if [ -f "${HOME}/google-cloud-sdk/path.zsh.inc" ]; then
    echo "Found google-cloud-sdk."
    source "${HOME}/google-cloud-sdk/path.zsh.inc"
fi

export NVM_DIR="$HOME/.nvm"

eval "$(pyenv init -)"

[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

export CROPASS_PASS_DIR="$HOME/Dropbox/cropass-encrypted-passwords"

export CCACHE_DIR=$HOME/.ccache
export CCACHE_TEMPDIR=$HOME/.ccache

export OCAMLPARAM="_,bin-annot=1"
export OPAMKEEPBUILDDIR=1

export MANPAGER='nvim +Man!'
# I don't use C-w
stty werase undef

export XDG_CACHE_HOME=$HOME/.cache
export XDG_CONFIG_HOME=$HOME/.config
export XDG_DATA_HOME=$HOME/.local/share

########## Environment variables end ##########

########## tmux start ##########

function tnew(){
    local current_dir=$(pwd)
    local d=$(basename "${current_dir}" | tr . _)
    tmux new -s ${d}
}

alias ta='tmux a -d -t'
alias tls='tmux ls'
alias tkill='tmux kill-session -t'

########## tmux end ##########

########## git start ##########

alias gl="git log --graph --date=short --pretty=\"format:%C(yellow)%h %C(cyan)%ad %C(green)%an%Creset%x09%s %C(red)%d%Creset\""
alias gs="git status"
alias gg="git log --graph --all --decorate=full"
alias gc="git checkout -b"
alias gf="git fetch --all"
alias gd="git diff"
alias gr="git remote -v"
alias gcm="git commit -m"
alias gwip="git add -u && git commit -m \"WIP\" && git push origin `git rev-parse --abbrev-ref HEAD`"

function glm(){
    local is_master=$(git branch | grep master)
    local is_main=$(git branch | grep main)
    local master_or_main=""
    if [ ! -z "${is_master}" ]; then
        master_or_main=master
    elif [ ! -z "${is_main}" ]; then
        master_or_main=main
    else
        echo "Cannot find master or main branch."
        return
    fi

    local backup_branch=${master_or_main}-backup-$(date "+%Y%m%d%M%S")
    git checkout ${master_or_main}
    git switch --create ${backup_branch}
    git branch -D ${master_or_main}
    git fetch upstream ${master_or_main}
    git switch --create ${master_or_main} upstream/${master_or_main}
}

function grm(){
    git fetch --all
    local is_master=$(git branch --all | grep upstream/master)
    local is_main=$(git branch --all | grep upstream/main)
    if [ ! -z "${is_master}" ]; then
        git rebase upstream/master
    elif [ ! -z "${is_main}" ]; then
        git rebase upstream/main
    else
        echo "Cannot find master or main branch."
    fi
}

function gp(){
    git push origin $(git rev-parse --abbrev-ref HEAD)
}

function gpf(){
    git push origin $(git rev-parse --abbrev-ref HEAD) --force-with-lease --force-if-includes
}

function git-backup(){
    local branchname=$(git rev-parse --abbrev-ref HEAD)
    local backup=${branchname}-backup-$(date "+%Y%m%d%M%S")
    git branch ${backup}
    git push origin ${backup}
    git checkout ${backup}
    git branch -D ${branchname}
}

########## git end ##########

########## PROMPT start ##########
# vcs_info
autoload -Uz vcs_info
autoload -Uz add-zsh-hook

zstyle ':vcs_info:*' formats '%F{red}(%b)%f'
zstyle ':vcs_info:*' actionformats '%F{red}(%b|%a)%f'

function _update_vcs_info_msg() {
    LANG=en_US.UTF-8 vcs_info
}
add-zsh-hook precmd _update_vcs_info_msg

# Prompt

function preexec() {
  timer=$(($(date +%s%0N)/1000000))
}

setopt PROMPT_SUBST
PROMPT='%{${fg[green]}%}[@%m]%{${reset_color}%}${vcs_info_msg_0_}${PWD/#$HOME/~}
$ '

function precmd() {
  if [ $timer ]; then
    now=$(($(date +%s%0N)/1000000))
    elapsed=$(($now-$timer))

    if [ $elapsed -ge 10000 ]
    then
        echo "${elapsed}ms"
    fi
    unset timer
  fi
}


########## PROMPT end ##########

########## zsh miscellaneous start ###########

# Enable color
autoload -Uz colors
colors

# Vim-style key-binding
bindkey -v

# History setting
HISTFILE=~/.zsh_history
HISTSIZE=1000000
SAVEHIST=1000000

# 単語の区切り文字を指定する
autoload -Uz select-word-style
select-word-style default
# ここで指定した文字は単語区切りとみなされる
# / も区切りと扱うので、^W でディレクトリ１つ分を削除できる
zstyle ':zle:*' word-chars " /=;@:{},|"
zstyle ':zle:*' word-style unspecified

########################################
# 補完
# 補完機能を有効にする
# autoload -Uz compinit
# compinit -u

# 補完で小文字でも大文字にマッチさせる
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'

# ../ の後は今いるディレクトリを補完しない
zstyle ':completion:*' ignore-parents parent pwd ..

# sudo の後ろでコマンド名を補完する
zstyle ':completion:*:sudo:*' command-path /usr/local/sbin /usr/local/bin \
                   /usr/sbin /usr/bin /sbin /bin /usr/X11R6/bin

# ps コマンドのプロセス名補完
zstyle ':completion:*:processes' command 'ps x -o pid,s,args'

# オプション
# 日本語ファイル名を表示可能にする
setopt print_eight_bit

# beep を無効にする
setopt no_beep

# フローコントロールを無効にする
setopt no_flow_control

# Ctrl+Dでzshを終了しない
setopt ignore_eof

# '#' 以降をコメントとして扱う
setopt interactive_comments

# ディレクトリ名だけでcdする
setopt auto_cd

# cd したら自動的にpushdする
setopt auto_pushd
# 重複したディレクトリを追加しない
setopt pushd_ignore_dups

# 同時に起動したzshの間でヒストリを共有する
setopt share_history

# 同じコマンドをヒストリに残さない
setopt hist_ignore_all_dups

# スペースから始まるコマンド行はヒストリに残さない
setopt hist_ignore_space

# ヒストリに保存するときに余分なスペースを削除する
setopt hist_reduce_blanks

# 高機能なワイルドカード展開を使用する
setopt extended_glob

# キーバインド

# ^R で履歴検索をするときに * でワイルドカードを使用出来るようにする
bindkey '^R' history-incremental-pattern-search-backward

# vim:set ft=zsh:

########## zsh miscellaneous end ###########

########## alias start ##########

alias v='nvim'

alias la='ls -al'
alias ll='ls -l'

alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

alias mkdir='mkdir -p'

alias find-grep='find . -name "*" -type f | xargs grep'

alias remove-color="sed 's/\x1b\[[0-9;]*m//g'"

# Enable alias after sudo
alias sudo='sudo '

# Global alias
# alias -g L='| less'
# alias -g G='| grep'
alias -g ...='../..'
alias -g ....='../../..'

########## alias end ##########

############ Copy to stdout to clipboard start ############
# mollifier delta blog : http://mollifier.hatenablog.com/entry/20100317/p1

if which pbcopy >/dev/null 2>&1 ; then
    # Mac
    alias -g C='| pbcopy'
elif which xsel >/dev/null 2>&1 ; then
    # Linux
    alias -g C='| xsel --input --clipboard'
elif which putclip >/dev/null 2>&1 ; then
    # Cygwin
    alias -g C='| putclip'
fi

############ Copy to stdout to clipboard end ############

########### OS specific settings start ##########

case ${OSTYPE} in
    darwin*)
        export CLICOLOR=1
        alias ls='ls -G -F'
        ;;
    linux*)
        alias ls='ls -F --color=auto'
        ;;
esac

########### OS specific settings end ##########

########### Haskell start ##########

function haskell-purify-force(){
    if [ $# -ne 1 ]; then
        echo "You need to give a haskell source code."
    else
        hlint --refactor --refactor-options='-i' $1
        hlint --refactor --refactor-options='-i' $1
        hlint --refactor --refactor-options='-i' $1
        stylish-haskell --inplace $1
    fi
}

########### Haskell end ##########

########### OCaml start ##########

alias ocaml='ledit ocaml'
. $HOME/.opam/opam-init/init.zsh > /dev/null 2> /dev/null || true

fpath=(~/.zsh/completion $fpath)

########### OCaml end ##########

########## fzf start ########## 
# C-v checkout branch
# C-r Search history
# C-t Search files under the current directory

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

export FZF_TMUX=1
export FZF_TMUX_OPTS="--reverse -p95%"

function fzf-checkout-branch() {
  local branches branch
  branches=$(git branch --all | sed -e 's/\(^\* \|^  \)//g' | cut -d " " -f 1)
  branch=$(echo "$branches" | fzf-tmux -p 95% --reverse --preview "git show --color=always {}")

  local_branch=$(echo "$branch" | sed -e "s:^remotes/::")
  create_origin_branch=$(echo "$branch" | sed -e "s:^remotes/origin/::")

  if [[ "${branch}" == "${local_branch}" ]]
  then
    git switch ${branch}
  elif [[ "${branch}" == "${create_origin_branch}" ]]
  then
    git switch --detach ${branch}
  else
    git switch --create ${create_origin_branch} ${branch}
  fi
}
zle     -N   fzf-checkout-branch
bindkey "^v" fzf-checkout-branch

fzf-z-search() {
    local res=$(z | sort -rn | cut -c 12- | fzf-tmux -p 95% --reverse)
    if [ -n "$res" ]; then
        BUFFER+="cd $res"
        zle accept-line
    else
        return 1
    fi
}

zle -N fzf-z-search
bindkey '^z' fzf-z-search

fzf-f-locate() {
    local res=$(find / 2>/dev/null | fzf-tmux -p 95% --reverse)
    if [ -n "$res" ]; then
        BUFFER+="$res"
    else
        return 1
    fi
}

zle -N fzf-f-locate
bindkey '^f' fzf-f-locate

function ghq-fzf() {
  local src=$(ghq list | fzf-tmux -p 95% --preview "ls -la $(ghq root)/{} | tail -n+4 | awk '{print \$9\"/\"\$6\"/\"\$7 \" \" \$10}'")
  if [ -n "$src" ]; then
    BUFFER="cd $(ghq root)/$src"
    zle accept-line
  fi
  zle -R -c
}
zle -N ghq-fzf
bindkey '^]' ghq-fzf

function ghsearch-fzf() {
  ghsearch emit | fzf-tmux -p 95% --reverse --preview 'ghsearch preview {} {q}' --preview-window=,~2 | xargs -I{} ghsearch open {}
}
zle -N ghsearch-fzf
bindkey '^g' ghsearch-fzf

########## fzf end ########## 

########## wonnix start ##########

viewonnx() {
  onnx=${1?Missing ONNX file path.}

  if [ ! -f "${onnx}" ]; then
    printf "File not foud: \e[1m${onnx}\x1B[0m\n"
    return
  fi

  if [ ! -e $HOME/onnx2html.py ]; then
    curl https://raw.githubusercontent.com/shinh/test/master/onnx2html.py > $HOME/onnx2html.py
  fi

  tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/onnx2html.XXXXXXXXXX")

  python3 $HOME/onnx2html.py "${onnx}" "${tmpdir}/onnx.html"

  if [ -x "$(command -v w3m)" ]; then
    w3m "${tmpdir}/onnx.html"
  fi

  printf "Generated html at \e[4m\e[1m${tmpdir}/onnx.html\x1B[0m\n"
}

########## wonnix end ##########

########## Load machine specific settings start ##########

[ -f ~/.machine_specific.zsh ] && source ~/.machine_specific.zsh

########## Load machine specific settings end ##########

########## Remove duplicated entry from PATH start ###########

typeset -U PATH

########## Remove duplicated entry from PATH end ###########

########## Remove /bin from PATH start ############

export PATH=$(echo $PATH | sed -e "s|:/bin:|:|g"):/bin

########## Remove /bin from PATH end ############

########## ssh-agent start #########

if [[ "$(gpg --list-secret-keys | grep 3FB4269CA58D57F0326C1F7488737135568C1AC5 | wc -l)" == "1" ]]; then
    echo "Setup ssh-agent with gpg-agent because I found the gpg key."
    # gpg --export-ssh-key <key-id> to get the public key
    gpg-agent --daemon --enable-ssh-support
    export SSH_AUTH_SOCK="$(gpgconf --list-dirs agent-ssh-socket)"
else
    echo "Try setting up ssh-agent with ssh identification file because I cannot find the gpg key."
    if [[ -f ~/.ssh/id_rsa ]]; then
        echo "Setup ssh-agent with ~/.ssh/id_rsa."
        eval $(ssh-agent -s) > /dev/null
        ssh-add ~/.ssh/id_rsa
    elif [[ -f ~/.ssh/id_ed25519 ]]; then
        echo "Setup ssh-agent with ~/.ssh/id_ed25519."
        eval $(ssh-agent -s) > /dev/null
        ssh-add ~/.ssh/id_ed25519
    else
        echo "No ssh identification file found. Failed to start ssh-agent."
    fi
fi

# https://wiki.archlinux.org/title/GnuPG#Configure_pinentry_to_use_the_correct_TTY
export GPG_TTY=$(tty)
gpg-connect-agent updatestartuptty /bye >/dev/null

########## ssh-agent end #########

########## ssh-agent and tmux start #########

agent="$HOME/.ssh/agent"
if [ -S "$SSH_AUTH_SOCK" ]; then
    case $SSH_AUTH_SOCK in
    /tmp/*/agent.[0-9]*)
        ln -snf "$SSH_AUTH_SOCK" $agent && export SSH_AUTH_SOCK=$agent
    esac
elif [ -S $agent ]; then
    export SSH_AUTH_SOCK=$agent
else
    echo "no ssh-agent"
fi

########## ssh-agent and tmux end #########

# ########## zenlog start ##########
# 
# export ZENLOG_SRC_DIR=$HOME/src/zenlog/
# export ZENLOG_DIR=/tmp/zenlog-dir/
# 
# # Set up zenlog.
# # Note the following command does *not* start a zenlog session.
# # Type "zenlog" manually to start one, or change your terminal app's setting
# # to start zenlog instead of your login shell.
# #
# # Uncomment the following line to suppress zenlog default prompt.
# # ZENLOG_NO_DEFAULT_PROMPT=1
# #
# # Uncomment the following line to suppress zenlog default key bindings.
# # ZENLOG_NO_DEFAULT_BINDING=1
# 
# # Open log files in this command.
# ZENLOG_VIEWER=nvim
# 
# # Open raw log in this command. (Requires A2H.)
# ZENLOG_RAW_VIEWER=google-chrome
# 
# . <(zenlog basic-zsh-setup)
# 
# zenlog_gh_gist_last_cmd() {
#     if [[ ! -e ${HOME}/akawashiro-pfn-tools/make-akawashiro-gist.sh ]]; then
#         echo "Cannot find make-akawashiro-gist.sh"
#         return
#     fi
#     ${HOME}/akawashiro-pfn-tools/make-akawashiro-gist.sh $(realpath ${ZENLOG_DIR}/S)
# }
# 
# zle -N zenlog_gh_gist_last_cmd
# bindkey '^[5' zenlog_gh_gist_last_cmd
# 
# if which zenlog; then
#     zenlog
# fi
# 
# ########## zenlog end ##########

########## renlog start #########

if which renlog > /dev/null 2>&1; then
    if [[ "$(ps -o comm= -p $PPID)" != "renlog" ]]; then
        renlog_dir=$(mktemp -d /tmp/renlog.XXXXXX)
        exec renlog --log-level info log --renlog-dir ${renlog_dir} --cmd 'zsh -l'
    else
        eval "$(renlog show-zsh-rc)"
    fi
fi

renlog_view_last_cmd() {
    local last_log_file=$(cat ${RENLOG_LAST_LOG_FILE})
    if [ -f "${last_log_file}" ]; then
        nvim "${last_log_file}"
    else
        echo "No log file found."
    fi
}

zle -N renlog_view_last_cmd
bindkey '^[1' renlog_view_last_cmd

renlog_gist_last_cmd() {
    local last_log_file=$(cat ${RENLOG_LAST_LOG_FILE})
    if [[ ! -e ${HOME}/akawashiro-pfn-tools/make-akawashiro-gist.sh ]]; then
        echo "Cannot find make-akawashiro-gist.sh"
        return
    fi
    ${HOME}/akawashiro-pfn-tools/make-akawashiro-gist.sh $(realpath ${last_log_file})
}

zle -N renlog_gist_last_cmd
bindkey '^[2' renlog_gist_last_cmd

########## renlog end #########

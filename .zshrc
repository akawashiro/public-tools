########## Environment variables start ##########

export LANG=en_US.UTF-8

export PATH=/var/lib/snapd/snap/bin:$PATH
export PATH=$HOME/.cabal/bin:$PATH
export PATH=$HOME/.local/bin:$PATH
export PATH=$HOME/pfn-tools:$PATH
export PATH=$HOME/public-tools:$PATH
export PATH=$HOME/tools:$PATH
export PATH=$HOME/.cargo/bin:$PATH
export PATH=$HOME/.go/bin:$PATH
export PATH="$GOPATH/.bin:$PATH"
export PATH=$HOME/.gem/ruby/2.7.0/bin:$PATH
export PATH=$HOME/.fzf/bin:$PATH
# export PATH="$PYENV_ROOT/bin:$PATH"

# export PYENV_ROOT="$HOME/.pyenv"
# eval "$(pyenv init -)"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

export GOPATH="$HOME/.go"

export CROPASS_PASS_DIR="$HOME/Dropbox/cropass-encrypted-passwords"

export CCACHE_DIR=$HOME/.ccache
export CCACHE_TEMPDIR=$HOME/.ccache

export OCAMLPARAM="_,bin-annot=1"
export OPAMKEEPBUILDDIR=1

export MANPAGER='nvim +Man!'
# I don't use C-w
stty werase undef

########## Environment variables end ##########

########## zplug start ##########

source ~/.zplug/init.zsh

zplug "rupa/z", use:z.sh

if ! zplug check --verbose; then
    printf "Install? [y/N]: "
    if read -q; then
        echo; zplug install
    fi
fi

zplug load --verbose

########## zplug end ##########

########## tmux start ##########

function tnew(){
    local c=$(pwd)
    local d=$(basename "${c}" | tr . _)
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
alias gwip="git add -u && git commit -m \"WIP\" && git push origin `git rev-parse --abbrev-ref HEAD`"
alias gist="gh gist create --public"

function glm(){
    git fetch --all
    local is_master=$(git branch | grep master)
    local is_main=$(git branch | grep main)
    if [ ! -z "${is_master}" ]; then
        git checkout master
        git merge upstream/master
    elif [ ! -z "${is_main}" ]; then 
        git checkout main
        git merge upstream/main
    else
        echo "Cannot find master or main branch."
        exit 1
    fi
}

function gp(){
    git push origin $(git rev-parse --abbrev-ref HEAD)
}

function git-backup(){
    local branchname=$(git rev-parse --abbrev-ref HEAD)
    git branch ${branchname}-backup-$(date "+%Y%m%d%M%S")
    git push origin ${branchname}-backup-$(date "+%Y%m%d%M%S")
    git checkout ${branchname}
}

########## git end ##########

########## kubectl start ##########

alias kgp="pf kubectl get pod"
alias kgj="pf kubectl get job"
alias kd="pf kubectl describe"
alias kdp="pf kubectl describe pod"
alias kdj="pf kubectl describe job"
alias klf="pf kubectl logs -f"
alias delete-succeeded-jobs="kubectl delete jobs --field-selector status.successful=1"

########## kubectl end ##########

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
> '

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


########################################
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

########################################
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

# kubernetes
# [ -f $HOME/tools/.kubectl_aliases ] && source $HOME/tools/.kubectl_aliases
# source <(kubectl completion zsh)

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

########################################
# OS specific settings
case ${OSTYPE} in
    darwin*)
        export CLICOLOR=1
        alias ls='ls -G -F'
        ;;
    linux*)
        alias ls='ls -F --color=auto'
        ;;
esac

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
# C-b checkout branch
# C-r Search history
# C-t Search files under the current directory

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

export FZF_TMUX=1
export FZF_TMUX_OPTS="--reverse -p80%"

function fzf-checkout-branch() {
  local branches branch
  branches=$(git branch --all | sed -e 's/\(^\* \|^  \)//g' | cut -d " " -f 1)
  branch=$(echo "$branches" | fzf-tmux -p 80% --reverse --preview "git show --color=always {}")

  local_branch=$(echo "$branch" | sed -e "s:^remotes/::")
  create_origin_branch=$(echo "$branch" | sed -e "s:^remotes/origin/::")

  if [[ "${branch}" == "${local_branch}" ]]
  then
    git switch ${branch}
  elif [[ "${branch}" == "${create_origin_branch}" ]]
  then
    git switch --detach ${branch}
  else
    git switch --create ${create_origin_branch} --track=${branch} ${branch}
  fi
}
zle     -N   fzf-checkout-branch
bindkey "^b" fzf-checkout-branch

fzf-z-search() {
    local res=$(z | sort -rn | cut -c 12- | fzf-tmux -p 80% --reverse)
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
    local res=$(find / 2>/dev/null | fzf-tmux -p 80% --reverse)
    if [ -n "$res" ]; then
        BUFFER+="$res"
    else
        return 1
    fi
}

zle -N fzf-f-locate
bindkey '^f' fzf-f-locate

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

########## zenlog start ##########

export ZENLOG_SRC_DIR=$HOME/src/zenlog/

# Set up zenlog.
# Note the following command does *not* start a zenlog session.
# Type "zenlog" manually to start one, or change your terminal app's setting
# to start zenlog instead of your login shell.
#
# Uncomment the following line to suppress zenlog default prompt.
# ZENLOG_NO_DEFAULT_PROMPT=1
#
# Uncomment the following line to suppress zenlog default key bindings.
# ZENLOG_NO_DEFAULT_BINDING=1
#

# Open log files in this command.
ZENLOG_VIEWER=nvim

# Open raw log in this command. (Requires A2H.)
ZENLOG_RAW_VIEWER=google-chrome

. <(zenlog basic-zsh-setup)

########## zenlog end ##########

########## Load machine specific settings start ##########

[ -f ~/.machine_specific.zsh ] && source ~/.machine_specific.zsh

########## Load machine specific settings end ##########

# Remove duplicated entry from PATH
typeset -U PATH

# Remove /bin from PATH
export PATH=$(echo $PATH | sed -e "s|:/bin:|:|g"):/bin

if which zenlog; then
    zenlog
fi

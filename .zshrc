########################################
# Environment variables
export LANG=en_US.UTF-8

export PATH=$HOME/.cabal/bin:$PATH
export PATH=$HOME/.local/bin:$PATH
export PATH=$HOME/.cargo/bin:$PATH
export PATH=$HOME/.go/bin:$PATH
export PATH="$GOPATH/.bin:$PATH"
export PATH=$HOME/.gem/ruby/2.7.0/bin:$PATH

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

export GOPATH="$HOME/.go"

export CROPASS_PASS_DIR="$HOME/Dropbox/cropass-encrypted-passwords"

export CCACHE_DIR=$HOME/.ccache
export CCACHE_TEMPDIR=$HOME/.ccache

export OCAMLPARAM="_,bin-annot=1"
export OPAMKEEPBUILDDIR=1

# tmux auto rename

function tnew(){
    local c=`pwd`
    local d=`basename $c`
    tmux new -s ${d}
}

function ghist(){
    git diff HEAD~$1
}

# 色を使用出来るようにする
autoload -Uz colors
colors

# vim 風キーバインドにする
bindkey -v

# ヒストリの設定
HISTFILE=~/.zsh_history
HISTSIZE=1000000
SAVEHIST=1000000

# プロンプト
# 1行表示
# PROMPT="%~ %# "
# 2行表示
PROMPT="%{${fg[green]}%}[%n@%m]%{${reset_color}%} %* %~
%# "


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
autoload -Uz compinit
compinit -i

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
# vcs_info
autoload -Uz vcs_info
autoload -Uz add-zsh-hook

zstyle ':vcs_info:*' formats '%F{green}(%s)-[%b]%f'
zstyle ':vcs_info:*' actionformats '%F{red}(%s)-[%b|%a]%f'

function _update_vcs_info_msg() {
    LANG=en_US.UTF-8 vcs_info
    RPROMPT="${vcs_info_msg_0_}"
}
add-zsh-hook precmd _update_vcs_info_msg


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

########################################
# エイリアス

alias nv='nvim'

alias la='ls -al'
alias ll='ls -l'

alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

alias mkdir='mkdir -p'

alias ta='tmux a -t'
alias tls='tmux ls'
alias tkill='tmux kill-session -t'
alias find-grep='find . -name "*" -type f | xargs grep'

alias glog="git log --graph --date=short --pretty=\"format:%C(yellow)%h %C(cyan)%ad %C(green)%an%Creset%x09%s %C(red)%d%Creset\""
alias gsta="git status"
alias ggraph="git log --graph --all --decorate=full"
alias gdiff="ydiff -s -w0"

# sudo の後のコマンドでエイリアスを有効にする
alias sudo='sudo '

# グローバルエイリアス
alias -g L='| less'
alias -g G='| grep'
alias -g ...='../..'
alias -g ....='../../..'

# kubernetes
# [ -f $HOME/tools/.kubectl_aliases ] && source $HOME/tools/.kubectl_aliases
# source <(kubectl completion zsh)

# C で標準出力をクリップボードにコピーする
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

########################################
# OS 別の設定
case ${OSTYPE} in
    darwin*)
        #Mac用の設定
        export CLICOLOR=1
        alias ls='ls -G -F'
        ;;
    linux*)
        #Linux用の設定
        alias ls='ls -F --color=auto'
        ;;
esac

# vim:set ft=zsh:

alias ocaml='ledit ocaml'


alias tl='tw -tl'

function ggl() {
  command w3m "www.google.co.jp/search?q=${1}"
}

# alias for pip10. this line will be removed after upgrade of ubuntu.
# alias pip='python -m pip'

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

# OPAM configuration
. $HOME/.opam/opam-init/init.zsh > /dev/null 2> /dev/null || true

fpath=(~/.zsh/completion $fpath)

if [ -S "$SSH_AUTH_SOCK" ]; then
  if ! ssh-add -l > /dev/null; then
    ssh-add "$HOME/.ssh/id_rsa"
  fi
fi

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh
